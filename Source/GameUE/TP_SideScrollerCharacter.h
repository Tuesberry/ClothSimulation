// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "ClothSolver.h"
#include "CollisionInfoManager.h"
#include "ProceduralMeshComponent.h"
#include "Mesh.h"
#include "TP_SideScrollerCharacter.generated.h"

UCLASS(config = Game)
class ATP_SideScrollerCharacter : public ACharacter
{
	GENERATED_BODY()

		/** Side view camera */
		UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
		class UCameraComponent* SideViewCameraComponent;

	/** Camera boom positioning the camera beside the character */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
		class USpringArmComponent* CameraBoom;

protected:

	// variable
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool wasPunchAttackUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool wasPunch_R_AttackUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool wasKickAttackUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool wasKick_L_AttackUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool wasSkillAttackUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool wasFarAttackUsed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ComboAttack")
		bool canUsePunchCombo;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ComboAttack")
		bool canUseKickCombo;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ComboAttack")
		bool canUsePunchKickCombo;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ComboAttack")
		bool wasPunchComboUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ComboAttack")
		bool wasKickComboUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ComboAttack")
		bool wasPunchKickComboUsed;



	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Move")
		bool wasCrouchUsed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avoid")
		bool wasBlockUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avoid")
		bool wasHurtUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avoid")
		bool wasHurtComboUsed;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Avoid")
		bool wasDeadUsed;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Health")
		float playerHealth;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Health")
		float playerStamina;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Attacks")
		bool isAttack;

	bool canMove;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MyVariables")
		bool isKeyPressed;



	/** Called for side to side input */

	void MoveRight(float Val);
	void MoveLeft(float Val);

	/** Handle touch inputs. */
	void TouchStarted(const ETouchIndex::Type FingerIndex, const FVector Location);

	/** Handle touch stop event. */
	void TouchStopped(const ETouchIndex::Type FingerIndex, const FVector Location);

	// APawn interface
	virtual void SetupPlayerInputComponent(class UInputComponent* InputComponent) override;
	// End of APawn interface

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Hitbox")
		AActor* hurtbox;

	// Combo
	void attackPunch_RPress();
	void attackKickPress();

	// Attack
	void attackPunch();
	void attackPunch_R();
	void attackKick();
	void attackKick_L();
	void attackFar();
	void attackSkill();

	//Move
	void myJump();
	void moveCrouch();
	void stopCrouch();

	//Avoid
	void avoidBlock();
	void stopBlock();

	void animHurt();
	void animHurtCombo();
	void animDead();

	//Damage
	void TakeDamage(float _damageAmount);
	void TakeStamina(float _staminaAmount);



public:
	ATP_SideScrollerCharacter();

	/** Returns SideViewCameraComponent subobject **/
	FORCEINLINE class UCameraComponent* GetSideViewCameraComponent() const { return SideViewCameraComponent; }
	/** Returns CameraBoom subobject **/
	FORCEINLINE class USpringArmComponent* GetCameraBoom() const { return CameraBoom; }

#pragma region JH_SIM

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	void GetTriangle_FromCloth(USkeletalMeshComponent* model);
	void GetTriangle_FromClothProcedural(UProceduralMeshComponent* model);
	void CopySkeletalMeshToProcedural(USkeletalMeshComponent* SkeletalMeshComponent, int32 LODndex, UProceduralMeshComponent* ProcMeshComponent);
	void UpdateCollider();
public:
	glm::mat4 UE_MatrixTo_GL(FTransform t);
	void ApplyTransform(vector<glm::vec3>& positions, glm::mat4 transform)
	{
		for (int i = 0; i < positions.size(); i++)
		{
			positions[i] = transform * glm::vec4(positions[i], 1.0);
		}
	}

	void GenerateStretch(const vector<glm::vec3>& positions)
	{
		int res = 50;
		auto VertexAt = [this](int x, int y) {
			return x * (50 + 1) + y;
		};
		auto DistanceBetween = [&positions](int idx1, int idx2) {
			return glm::length(positions[idx1] - positions[idx2]);
		};
		for (int x = 0; x < res + 1; x++)
		{
			for (int y = 0; y < res + 1; y++)
			{
				int idx1, idx2;

				if (y != res)
				{
					idx1 = VertexAt(x, y);
					idx2 = VertexAt(x, y + 1);
					solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
				}

				if (x != res)
				{
					idx1 = VertexAt(x, y);
					idx2 = VertexAt(x + 1, y);
					solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
				}

				if (y != res && x != res)
				{
					idx1 = VertexAt(x, y);
					idx2 = VertexAt(x + 1, y + 1);
					solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));

					idx1 = VertexAt(x, y + 1);
					idx2 = VertexAt(x + 1, y);
					solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
				}
			}
		}
	}

	void GenerateBending(const vector<unsigned int>& indices)
	{
		// HACK: not for every kind of mesh
		for (int i = 0; i < indices.size(); i += 6)
		{
			int idx1 = indices[i];
			int idx2 = indices[i + 1];
			int idx3 = indices[i + 2];
			int idx4 = indices[i + 5];

			// TODO: calculate angle
			float angle = 0;
			solver->AddBend(m_indexOffset + idx1, m_indexOffset + idx2, m_indexOffset + idx3, m_indexOffset + idx4, angle);
		}
	}

	void GenerateAttach(const vector<glm::vec3>& positions)
	{
		for (int slotIdx = 0; slotIdx < m_attachedIndices.size(); slotIdx++)
		{
			int particleID = m_attachedIndices[slotIdx];
			glm::vec3 slotPos = positions[particleID];
			solver->AddAttachSlot(slotPos);
			for (int i = 0; i < positions.size(); i++)
			{
				float restDistance = glm::length(slotPos - positions[i]);
				solver->AddAttach(m_indexOffset + i, slotIdx, restDistance);
			}
			//m_solver->AddAttach(idx, positions[idx], 0);
		}
	}
	shared_ptr<JH::Mesh> GenerateClothMesh(int resolution)
	{
		vector<glm::vec3> vertices;
		vector<glm::vec3> normals;
		vector<glm::vec2> uvs;
		vector<unsigned int> indices;
		const float clothSize = 100.0f;

		for (int y = 0; y <= resolution; y++)
		{
			for (int x = 0; x <= resolution; x++)
			{
				vertices.push_back(clothSize * glm::vec3((float)x / (float)resolution - 0.5f, -(float)y / (float)resolution, 0));
				normals.push_back(glm::vec3(0, 0, 1));
				uvs.push_back(glm::vec2((float)x / (float)resolution, (float)y / (float)resolution));
			}
		}

		auto VertexIndexAt = [resolution](int x, int y) {
			return x * (resolution + 1) + y;
		};

		for (int x = 0; x < resolution; x++)
		{
			for (int y = 0; y < resolution; y++)
			{
				indices.push_back(VertexIndexAt(x, y));
				indices.push_back(VertexIndexAt(x + 1, y));
				indices.push_back(VertexIndexAt(x, y + 1));

				indices.push_back(VertexIndexAt(x, y + 1));
				indices.push_back(VertexIndexAt(x + 1, y));
				indices.push_back(VertexIndexAt(x + 1, y + 1));
			}
		}
		auto mesh = make_shared<JH::Mesh>(vertices, normals, uvs, indices);
		return mesh;
	}

	void StickClothToSocket();
	int m_resolution = 40;
	int m_indexOffset;
	vector<int> m_attachedIndices = { 0, 41 };
	float m_particleDiameter;

	UPROPERTY(BlueprintReadWrite, EditAnywhere)
		USkeletalMeshComponent* m_ClothMesh;
	UPROPERTY(BlueprintReadWrite, EditAnywhere)
		UProceduralMeshComponent* m_ClothProceduralMesh;
	UPROPERTY(BlueprintReadWrite, EditAnywhere)
		UMaterialInterface* m_ClothMaterialInterface;


	TArray<TTuple<FVector, FVector, FVector>> clothTriangles;
	TArray<TTuple<FVector, FVector, FVector>> clothProceduralTriangles;

	TArray<FVector> VerticesArray;
	TArray<FVector> Normals;
	TArray<FVector2D> UV;
	TArray<int32> Tris;
	TArray<FColor> Colors;
	TArray<FProcMeshTangent> Tangents;
	TArray<uint32> Indicies;
private:
	UCollisionInfoManager* infoMgr;
	ClothSolverGPU* solver;
	VtSimParams simParams;
	vector<Collider*> colliders;
	vector<Collider*> bf_colliders;
	shared_ptr<JH::Mesh> m_Mesh;

	vector<glm::vec3> clothPos;
	vector<glm::vec3> clothNormal;
	vector<glm::vec2> clothUV;
	vector<uint> ClothIdx;
#pragma endregion JH_SIM
};
