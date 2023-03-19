#pragma once
#include <iostream>

#include <glm/glm.hpp>
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
//#include <thrust/device_ptr.h>
//#include <thrust/transform.h>

#include "Mesh.h"
#include "ClothSolver.cuh"
#include "VtBuffer.h"
#include "SpatialHashGPU.h"

using namespace std;


class Collider
{
public:
	ColliderType type = ColliderType::Sphere;

	glm::vec3 pos;
	glm::vec3 rot;
	glm::vec3 scale;

	glm::vec3 lastPos;
	glm::vec3 velocity;
	glm::mat4 curTransform;
	glm::mat4 lastTransform;
	string name;
	Collider(ColliderType _type)
	{
		name = __func__;
		type = _type;
	}
	
};

class ClothSolverGPU 
{
public:
	ClothSolverGPU(vector<Collider*> ModelColliders, VtSimParams simParams, float deltaTime)
	{
		this->simParams = simParams;
		this->deltaTime = deltaTime;
		simParams.numParticles = 0;

		m_colliders = ModelColliders;
	}

	void FixedUpdate() 
	{
		UpdateColliders(m_colliders);
		Simulate();
	}

	void OnDestroy() 
	{
		positions.destroy();
		normals.destroy();
	}

	void Simulate()
	{
		//==========================
		// Prepare
		//==========================
		float frameTime = deltaTime;
		float substepTime = deltaTime / simParams.numSubsteps;

		//==========================
		// Launch kernel
		//==========================
		SetSimulationParams(&simParams);

		// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
		// This can produce unstable behavior, such as vertex flashing between two sides.
		// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
		CollideSDF(positions, sdfColliders, positions, (uint)sdfColliders.size(), frameTime);

		for (int substep = 0; substep < simParams.numSubsteps; substep++)
		{
			PredictPositions(predicted, velocities, positions, substepTime);
			
			if (simParams.enableSelfCollision)
			{
				if (substep % simParams.interleavedHash == 0)
				{
					m_spatialHash->Hash(predicted);
				}
				CollideParticles(deltas, deltaCounts, predicted, invMasses, m_spatialHash->neighbors, positions);
			}
			CollideSDF(predicted, sdfColliders, positions, (uint)sdfColliders.size(), substepTime);

			for (int iteration = 0; iteration < simParams.numIterations; iteration++)
			{
				SolveStretch(predicted, deltas, deltaCounts, stretchIndices, stretchLengths, invMasses, (uint)stretchLengths.size());
				SolveAttachment(predicted, deltas, deltaCounts, invMasses,
					attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, (uint)attachParticleIDs.size());
				//SolveBending(predicted, deltas, deltaCounts, bendIndices, bendAngles, invMasses, (uint)bendAngles.size(), substepTime);
				ApplyDeltas(predicted, deltas, deltaCounts);
			}

			Finalize(velocities, positions, predicted, substepTime);
		}

		//ComputeNormal(normals, positions, indices, (uint)(indices.size() / 3));

		//==========================
		// Sync
		//==========================
		cudaDeviceSynchronize();

	}
public:

	int AddCloth(shared_ptr<JH::Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
	{

		int prevNumParticles = simParams.numParticles;
		int newParticles = (int)mesh->vertices().size();

		// Set global parameters
		simParams.numParticles += newParticles;
		simParams.particleDiameter = particleDiameter;
		simParams.deltaTime = deltaTime;
		simParams.maxSpeed = 2 * particleDiameter / deltaTime * simParams.numSubsteps;

		// Allocate managed buffers
		positions.push_back(mesh->vertices());
		normals.push_back(mesh->normals());

		for (int i = 0; i < mesh->indices().size(); i++)
		{
			indices.push_back(mesh->indices()[i] + prevNumParticles);
		}

		velocities.push_back(newParticles, glm::vec3(0));
		predicted.push_back(newParticles, glm::vec3(0));
		deltas.push_back(newParticles, glm::vec3(0));
		deltaCounts.push_back(newParticles, 0);
		invMasses.push_back(newParticles, 1.0f);

		// Initialize buffer datas
		InitializePositions(positions, prevNumParticles, newParticles, modelMatrix);
		cudaDeviceSynchronize();

		// Initialize member variables
		m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, simParams.numParticles, simParams);
		m_spatialHash->SetInitialPositions(positions);

		return prevNumParticles;
	}

	void AddStretch(int idx1, int idx2, float distance)
	{
		stretchIndices.push_back(idx1);
		stretchIndices.push_back(idx2);
		stretchLengths.push_back(distance);
	}

	void AddAttachSlot(glm::vec3 attachSlotPos)
	{
		attachSlotPositions.push_back(attachSlotPos);
	}
	void InitAttachSlots(vector<glm::vec3> attachSlotPos)
	{
		attachSlotPositions.destroy();
		for (auto attachPos : attachSlotPos)
		{
			attachSlotPositions.push_back(attachPos);
		}
	}
	void AddAttach(int particleIndex, int slotIndex, float distance)
	{
		if (distance == 0) invMasses[particleIndex] = 0;
		attachParticleIDs.push_back(particleIndex);
		attachSlotIDs.push_back(slotIndex);
		attachDistances.push_back(distance);
	}

	void AddBend(uint idx1, uint idx2, uint idx3, uint idx4, float angle)
	{
		bendIndices.push_back(idx1);
		bendIndices.push_back(idx2);
		bendIndices.push_back(idx3);
		bendIndices.push_back(idx4);
		bendAngles.push_back(angle);
	}

	void UpdateColliders(vector<Collider*>& colliders)
	{
		sdfColliders.resize(colliders.size());

		for (int i = 0; i < colliders.size(); i++)
		{
			const Collider* c = colliders[i];
			if (c == nullptr) continue;
			SDFCollider sc;
			sc.type = c->type;
			sc.position = c->pos;
			sc.scale = c->scale;
			sc.curTransform = c->curTransform;
			sc.invCurTransform = glm::inverse(c->curTransform);
			sc.lastTransform = c->lastTransform;
			sc.deltaTime = deltaTime;
			sdfColliders[i] = sc;
		}
	}

public: // Sim buffers

	VtBuffer<glm::vec3> positions;
	VtBuffer<glm::vec3> normals;
	VtBuffer<uint> indices;

	VtBuffer<glm::vec3> velocities;
	VtBuffer<glm::vec3> predicted;
	VtBuffer<glm::vec3> deltas;
	VtBuffer<int> deltaCounts;
	VtBuffer<float> invMasses;

	VtBuffer<int> stretchIndices;
	VtBuffer<float> stretchLengths;
	VtBuffer<uint> bendIndices;
	VtBuffer<float> bendAngles;

	// Attach attachParticleIndices[i] with attachSlotIndices[i] 
	// where their expected distance is attachDistances[i]
	VtBuffer<int> attachParticleIDs;
	VtBuffer<int> attachSlotIDs;
	VtBuffer<float> attachDistances;
	VtBuffer<glm::vec3> attachSlotPositions;

	VtBuffer<SDFCollider> sdfColliders;
	

	VtSimParams simParams;
	float deltaTime;
private:

	shared_ptr<SpatialHashGPU> m_spatialHash;
	vector<Collider*> m_colliders;

};