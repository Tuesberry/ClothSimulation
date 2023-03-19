#pragma once
#include "common.cuh"

struct HashParams
{
	uint numObjects;
	uint maxNumNeighbors;
	float cellSpacing;
	float cellSpacing2;
	int tableSize;
	float particleDiameter2;
};

void HashObjects(
	uint* particleHash,
	uint* particleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint* neighbors,
	CONST(glm::vec3*) positions,
	CONST(glm::vec3*) originalPositions,
	const HashParams params);