// Copyright Epic Games, Inc. All Rights Reserved.

#include "TP_SideScrollerCharacter.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/InputComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "DrawDebugHelpers.h"
#include "Rendering/PositionVertexBuffer.h"
#include "RHICommandList.h"
#include "Components/SkinnedMeshComponent.h"
#include "RenderResource.h"
#include "Engine/SkeletalMeshSocket.h"
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>

ATP_SideScrollerCharacter::ATP_SideScrollerCharacter()
{
	// Set size for collision capsule
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

	// Don't rotate when the controller rotates.
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	// Create a camera boom attached to the root (capsule)
	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->SetUsingAbsoluteRotation(true); // Rotation of the character should not affect rotation of boom
	CameraBoom->bDoCollisionTest = false;
	CameraBoom->TargetArmLength = 300;
	CameraBoom->SocketOffset = FVector(0.f, 0.f, 75.f);
	CameraBoom->SetRelativeRotation(FRotator(0.f, 180.f, 0.f));

	// Create a camera and attach to boom
	SideViewCameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("SideViewCamera"));
	SideViewCameraComponent->SetupAttachment(CameraBoom, USpringArmComponent::SocketName);
	SideViewCameraComponent->bUsePawnControlRotation = false; // We don't want the controller rotating the camera

	// Configure character movement
	GetCharacterMovement()->bOrientRotationToMovement = true; // Face in the direction we are moving..
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 720.0f, 0.0f); // ...at this rotation rate
	GetCharacterMovement()->GravityScale = 2.f;
	GetCharacterMovement()->AirControl = 0.80f;
	GetCharacterMovement()->JumpZVelocity = 1000.f;
	GetCharacterMovement()->GroundFriction = 3.f;
	GetCharacterMovement()->MaxWalkSpeed = 600.f;
	GetCharacterMovement()->MaxFlySpeed = 600.f;

	// Note: The skeletal mesh and anim blueprint references on the Mesh component (inherited from Character) 
	// are set in the derived blueprint asset named MyCharacter (to avoid direct content references in C++)

	canMove = true;
	isAttack = false;
	isKeyPressed = false;

	hurtbox = nullptr;
	playerHealth = 1.00f;
	playerStamina = 0.00f;

	//ACharacter* myCharacter = UGameplayStatics::GetPlayerCharacter(GetWorld(), 0);
	//myCharacter->GetActorLocation();
	infoMgr = CreateDefaultSubobject<UCollisionInfoManager>("CollisionInfo");
	m_ClothMesh = CreateDefaultSubobject<USkeletalMeshComponent>("clothMesh");
	m_ClothMesh->AttachTo(RootComponent);
	m_ClothProceduralMesh = CreateDefaultSubobject<UProceduralMeshComponent>("ClothProceduralMesh");
	m_ClothProceduralMesh->AttachTo(RootComponent);
}
void ATP_SideScrollerCharacter::BeginPlay()
{
	Super::BeginPlay();
	//GetTriangle_FromCloth(m_ClothMesh);
	//CopySkeletalMeshToProcedural(m_ClothMesh, 0, m_ClothProceduralMesh);
	//m_ClothMesh->SetVisibility(false);

	solver = new ClothSolverGPU(colliders, simParams, 0.5f);
	m_Mesh = GenerateClothMesh(50);
	for (auto vert : m_Mesh->vertices())
	{
		VerticesArray.Add(FVector(vert.x, vert.y, vert.z));
		Colors.Add(FColor::Red);
	}
	for (auto vert : m_Mesh->normals())
	{
		Normals.Add(FVector(vert.x, vert.y, vert.z));
		glm::vec3 tangent;
		glm::vec3 c1 = glm::cross(vert, glm::vec3(0.0, 0.0, 1.0));
		glm::vec3 c2 = glm::cross(vert, glm::vec3(0.0, 1.0, 0.0));

		if (glm::length(c1) > glm::length(c2))
		{
			tangent = c1;
		}
		else
		{
			tangent = c2;
		}

		tangent = normalize(tangent);
		Tangents.Add((FProcMeshTangent(FVector(tangent.x, tangent.y, tangent.z), false)));
	}
	for (auto vert : m_Mesh->indices())
	{
		Tris.Add(vert);
	}
	for (auto uv : m_Mesh->uvs())
	{
		UV.Add(FVector2D(uv.x, uv.y));
	}

	float max = -1000;
	for (int i = 0; i < colliders.size(); i++)
	{
		if (colliders[i]->pos.z > max)
			max = colliders[i]->pos.z;
	}
	UE_LOG(LogTemp, Warning, L"Collider Max Z : %f", max);
	//m_Mesh = make_shared<JH::Mesh>(clothPos, clothNormal, clothUV, ClothIdx);


	glm::mat4 transform = glm::mat4(1.0f);
	transform = glm::scale(transform, glm::vec3(1.0f));
	transform = glm::translate(transform, glm::vec3(GetActorLocation().X + 100, GetActorLocation().Y - 300, GetActorLocation().Z + 300));

	m_particleDiameter = glm::length(m_Mesh->vertices()[0] - m_Mesh->vertices()[1]) * simParams.particleDiameterScalar;
	m_indexOffset = solver->AddCloth(m_Mesh, transform, m_particleDiameter);
	//sort(clothPos.begin(), clothPos.end(), [](glm::vec3& rhs, glm::vec3& lhs) 
	//	{
	//		if (rhs.x < lhs.x)
	//			return true;
	//		else if (rhs.x == lhs.x)
	//		{
	//			if (rhs.y < lhs.y)
	//				return true;
	//			else
	//				return false;
	//		}
	//		else return false;
	//	});

	GenerateStretch(m_Mesh->vertices());
	GenerateAttach(m_Mesh->vertices());
	GenerateBending(m_Mesh->indices());


}

// Called every frame
void ATP_SideScrollerCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	StickClothToSocket();
	solver->deltaTime = DeltaTime * 5;
	UpdateCollider();
	solver->Simulate();

	for (int i = 0; i < solver->positions.size(); i++)
	{
		glm::vec3 tVec = *(solver->positions.data() + i);

		VerticesArray[i] = (GetActorRotation() * -1).RotateVector((FVector(tVec.x, tVec.y, tVec.z)) - GetActorLocation());
	}
	for (int i = 0; i < m_Mesh->uvs().size(); i++)
	{
		glm::vec2 uvs = m_Mesh->uvs()[i];
		UV[i] = FVector2D(uvs.x, uvs.y);
	}
	for (int i = 0; i < solver->normals.size(); i++)
	{
		glm::vec3 tVec = *(solver->normals.data() + i);

		Normals[i] = (FVector(tVec.x, tVec.y, tVec.z));
		glm::vec3 tangent;
		glm::vec3 c1 = glm::cross(tVec, glm::vec3(0.0, 0.0, 1.0));
		glm::vec3 c2 = glm::cross(tVec, glm::vec3(0.0, 1.0, 0.0));

		if (glm::length(c1) > glm::length(c2))
		{
			tangent = c1;
		}
		else
		{
			tangent = c2;
		}

		tangent = normalize(tangent);
		Tangents[i] = (FProcMeshTangent(FVector(tangent.x, tangent.y, tangent.z), false));
	}
	for (int i = 0; i < solver->indices.size(); i++)
	{
		uint idx = *(solver->indices.data() + i);
		Tris[i] = idx;
	}

	/*for (int i = 0; i < solver->positions.size(); i+=100)
	{
		DrawDebugPoint(GetWorld(), VerticesArray[i], 10.0f, FColor::Blue, false, DeltaTime * 3);
	}*/
	//DrawDebugMesh(GetWorld(), VerticesArray, Tris, FColor::Blue, false, DeltaTime * 3);
	UE_LOG(LogTemp, Log, L"%f, %f, %f", VerticesArray[0].X, VerticesArray[0].Y, VerticesArray[0].Z);

	//for (int i = 0; i < solver->positions.size(); i++)
	//{
	//	glm::vec3 tVec = *(solver->positions.data() + i);
	//	DrawDebugPoint(GetWorld(), FVector(tVec.x, tVec.y, tVec.z), 10.0f, FColor(i / (float)solver->positions.size() * 256.0f, 255,255) , false, DeltaTime * 3);
	//} 
	m_ClothProceduralMesh->CreateMeshSection(0, VerticesArray, Tris, Normals, UV, Colors, Tangents, false);
	if (m_ClothMaterialInterface != nullptr)
	{
		m_ClothProceduralMesh->SetMaterial(0, m_ClothMaterialInterface);
	}
}

void ATP_SideScrollerCharacter::GetTriangle_FromCloth(USkeletalMeshComponent* cloth)
{

	FSkeletalMeshRenderData* data = cloth->GetSkeletalMeshRenderData();
	auto IdxBuffer = data->LODRenderData[0].MultiSizeIndexContainer.GetIndexBuffer();
	int32 IdxCount = IdxBuffer->Num();

	TArray<FMatrix> ref_to_local_m;
	TArray<FVector> Locations;

	cloth->GetCurrentRefToLocalMatrices(ref_to_local_m, 0);
	USkinnedMeshComponent::ComputeSkinnedPositions(cloth, Locations, ref_to_local_m, data->LODRenderData[0], data->LODRenderData[0].SkinWeightVertexBuffer);

	clothTriangles.Reserve(IdxCount / 3);

	////for every triangle
	for (int32 i = 0; i < IdxCount / 3; i++)
	{
		auto tuple = MakeTuple(GetTransform().TransformPosition(Locations[IdxBuffer->Get(i * 3)]),
			GetTransform().TransformPosition(Locations[IdxBuffer->Get(i * 3 + 1)]),
			GetTransform().TransformPosition(Locations[IdxBuffer->Get(i * 3 + 2)]));
		clothTriangles.Add(tuple);
	}
}

void ATP_SideScrollerCharacter::GetTriangle_FromClothProcedural(UProceduralMeshComponent* model)
{
	TArray<FProcMeshVertex> verts = model->GetProcMeshSection(0)->ProcVertexBuffer;
	TArray<uint32> idxs = model->GetProcMeshSection(0)->ProcIndexBuffer;

	clothProceduralTriangles.Reserve(idxs.Num() / 3);

	for (int32 i = 0; i < idxs.Num(); i += 3)
	{
		clothProceduralTriangles.Add(MakeTuple(verts[idxs[i]].Position, verts[idxs[i + 1]].Position, verts[idxs[i + 1]].Position));
	}
}

void ATP_SideScrollerCharacter::CopySkeletalMeshToProcedural(USkeletalMeshComponent* SkeletalMeshComponent, int32 LODIndex, UProceduralMeshComponent* ProcMeshComponent)
{
	FSkeletalMeshRenderData* SkMeshRenderData = SkeletalMeshComponent->GetSkeletalMeshRenderData();
	const FSkeletalMeshLODRenderData& DataArray = SkMeshRenderData->LODRenderData[LODIndex];
	FSkinWeightVertexBuffer& SkinWeights = *SkeletalMeshComponent->GetSkinWeightBuffer(LODIndex);

	//get num vertices
	int32 NumSourceVertices = DataArray.RenderSections[0].NumVertices;

	for (int32 i = 0; i < NumSourceVertices; i++)
	{
		//SkeletalMeshComponent->GetComponentTransform().TransformPosition(SkinnedVectorPos)
		// GetTransform().TransformPosition(SkinnedVectorPos);
		//get skinned vector positions
		FVector SkinnedVectorPos = USkeletalMeshComponent::GetSkinnedVertexPosition(SkeletalMeshComponent, i, DataArray, SkinWeights);
		//SkinnedVectorPos = SkeletalMeshComponent->GetComponentTransform().TransformPosition(SkinnedVectorPos);
		VerticesArray.Add(SkeletalMeshComponent->GetRelativeTransform().TransformPosition(SkinnedVectorPos));

		//Calc normals and tangents from the static version instead of the skeletal one
		FVector ZTangentStatic = DataArray.StaticVertexBuffers.StaticMeshVertexBuffer.VertexTangentZ(i);
		FVector XTangentStatic = DataArray.StaticVertexBuffers.StaticMeshVertexBuffer.VertexTangentX(i);

		//add normals from the static mesh version instead because using the skeletal one doesn't work right.
		Normals.Add(ZTangentStatic);

		//add tangents
		Tangents.Add(FProcMeshTangent(XTangentStatic, false));

		//get UVs
		FVector2D uvs = DataArray.StaticVertexBuffers.StaticMeshVertexBuffer.GetVertexUV(i, 0);
		UV.Add(uvs);

		//dummy vertex colors
		Colors.Add(FColor(150, 81, 168, 255));
	}


	//get index buffer
	DataArray.MultiSizeIndexContainer.GetIndexBuffer(Indicies);

	//iterate over num indices and add traingles
	for (int32 i = 0; i < Indicies.Num(); i++)
	{
		uint32 a = 0;
		a = Indicies[i];
		Tris.Add(a);
	}
	/*for (FVector& pos : VerticesArray)
	{
		UE_LOG(LogTemp, Log, L"%f %f %f", pos.X, pos.Y, pos.Z);
	}*/
	//Create the procedural mesh
	ProcMeshComponent->CreateMeshSection(0, VerticesArray, Tris, Normals, UV, Colors, Tangents, true);
}

void ATP_SideScrollerCharacter::UpdateCollider()
{
	infoMgr->CollisionSampling();
	bf_colliders = colliders;
	colliders.clear();
	Collider* col = nullptr;
	for (int i = 0; i < infoMgr->m_collisionArr.Num(); i++)
	{
		FCollisionInfo info = infoMgr->m_collisionArr[i];
		ColliderType type;
		//info.m_curTransform.GetTranslation().X
		switch (info.m_collisionType)
		{
		case ECollisionType::Sphere:
			DrawDebugSphere(GetWorld(), FVector(info.m_position.X, info.m_position.Y, info.m_position.Z), 10, 10, FColor::Red, false, 0.1f);
			type = ColliderType::Sphere;
			break;
		case ECollisionType::Box:
			type = ColliderType::Cube;
			DrawDebugBox(GetWorld(), FVector(info.m_position.X, info.m_position.Y, info.m_position.Z), FVector(info.m_extent.X, info.m_extent.Y, info.m_extent.Z), FColor::Green, false, 0.1f);
			break;
		case ECollisionType::Sphyl:
			DrawDebugCapsule(GetWorld(), FVector(info.m_position.X, info.m_position.Y, info.m_position.Z), info.m_height / 2, info.m_radius / 2, info.m_rotation, FColor::Blue, false, 0.1f);
			type = ColliderType::Sphere;
			break;
		}
		col = new Collider(type);
		col->type = type;
		col->pos = glm::vec3(info.m_position.X, info.m_position.Y, info.m_position.Z);
		//col->rot = glm::vec3(info.m_rotation.X, info.m_rotation.Y, info.m_rotation.Z);
		col->rot = glm::vec3(info.m_rotation.X, info.m_rotation.Y, info.m_rotation.Z);
		if (type == ColliderType::Sphere)
		{
			float colRad = max(info.m_radius, info.m_height);
			col->scale = glm::vec3(colRad);
		}
		else
		{
			col->scale = glm::vec3(info.m_extent.X, info.m_extent.Y, info.m_extent.Z);
		}
		glm::mat4 transform = glm::mat4(1.0);
		//glm::vec3 scale = glm::vec3(col->scale.x, col->scale.y, col->scale.z);
		//transform = glm::scale(transform, scale);
		//transform = glm::rotate(transform, col->rot.x, glm::vec3(1.0f, 0, 0));
		//transform = glm::rotate(transform, col->rot.y, glm::vec3(0, 1.0f, 0));
		//transform = glm::rotate(transform, col->rot.z, glm::vec3(0, 0, 1.0f));
		transform = glm::translate(transform, glm::vec3(GetActorLocation().X, GetActorLocation().Y, GetActorLocation().Z));
		FCollisionInfo lastColl = info;
		glm::mat4 ltransform = glm::mat4(1.0);
		if (infoMgr->m_BeforeCollisionArr.IsValidIndex(i))
		{
			lastColl = infoMgr->m_BeforeCollisionArr[i];
		}

		ltransform = glm::translate(ltransform, glm::vec3(lastColl.m_position.X, lastColl.m_position.Y, lastColl.m_position.Z));

		/*	ltransform = glm::scale(ltransform, scale);
			ltransform = glm::rotate(ltransform, col->rot.x, glm::vec3(1.0f, 0, 0));
			ltransform = glm::rotate(ltransform, col->rot.y, glm::vec3(0, 1.0f, 0));
			ltransform = glm::rotate(ltransform, col->rot.z, glm::vec3(0, 0, 1.0f));*/
		col->curTransform = transform;
		col->lastTransform = ltransform;
		colliders.push_back(col);
	}
	solver->UpdateColliders(colliders);
}

glm::mat4 ATP_SideScrollerCharacter::UE_MatrixTo_GL(FTransform t)
{
	glm::mat4 transform(1.0f);
	glm::vec3 scale = glm::vec3(t.GetScale3D().X, t.GetScale3D().Y, t.GetScale3D().Z);
	glm::vec3 rot = glm::vec3(t.GetRotation().X, t.GetRotation().Y, t.GetRotation().Z);
	glm::vec3 trans = glm::vec3(t.GetTranslation().X, t.GetTranslation().Y, t.GetTranslation().Z);
	transform = glm::scale(transform, scale);
	transform = glm::rotate(transform, rot.x, glm::vec3(1.0f, 0, 0));
	transform = glm::rotate(transform, rot.y, glm::vec3(0, 1.0f, 0));
	transform = glm::rotate(transform, rot.z, glm::vec3(0, 0, 1.0f));
	transform = glm::translate(transform, trans);
	return transform;
}

void ATP_SideScrollerCharacter::StickClothToSocket()
{
	vector<glm::vec3> tmpPos;
	FVector actorPos = GetActorLocation();
	FVector right = GetMesh()->GetSocketLocation("LeftSholder");
	FVector left = GetMesh()->GetSocketLocation("RightSholder");
	tmpPos.push_back(glm::vec3(right.X, right.Y, right.Z));
	tmpPos.push_back(glm::vec3(left.X, left.Y, left.Z));
	solver->InitAttachSlots(tmpPos);
}

//////////////////////////////////////////////////////////////////////////
// Input

void ATP_SideScrollerCharacter::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	// set up gameplay key bindings
	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ATP_SideScrollerCharacter::myJump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);
	PlayerInputComponent->BindAxis("MoveRight", this, &ATP_SideScrollerCharacter::MoveRight);
	PlayerInputComponent->BindAxis("MoveLeft", this, &ATP_SideScrollerCharacter::MoveLeft);
	PlayerInputComponent->BindAction("Crouch", IE_Pressed, this, &ATP_SideScrollerCharacter::moveCrouch);
	PlayerInputComponent->BindAction("Crouch", IE_Released, this, &ATP_SideScrollerCharacter::stopCrouch);
	PlayerInputComponent->BindAction("Block", IE_Pressed, this, &ATP_SideScrollerCharacter::avoidBlock);
	PlayerInputComponent->BindAction("Block", IE_Released, this, &ATP_SideScrollerCharacter::stopBlock);


	PlayerInputComponent->BindAction("Punch_R", IE_Pressed, this, &ATP_SideScrollerCharacter::attackPunch_RPress);
	PlayerInputComponent->BindAction("Kick", IE_Pressed, this, &ATP_SideScrollerCharacter::attackKickPress);

	PlayerInputComponent->BindAction("Punch", IE_Released, this, &ATP_SideScrollerCharacter::attackPunch);
	PlayerInputComponent->BindAction("Punch_R", IE_Released, this, &ATP_SideScrollerCharacter::attackPunch_R);
	PlayerInputComponent->BindAction("Kick", IE_Released, this, &ATP_SideScrollerCharacter::attackKick);
	PlayerInputComponent->BindAction("Kick_L", IE_Released, this, &ATP_SideScrollerCharacter::attackKick_L);
	PlayerInputComponent->BindAction("Far", IE_Released, this, &ATP_SideScrollerCharacter::attackFar);
	PlayerInputComponent->BindAction("Skill", IE_Released, this, &ATP_SideScrollerCharacter::attackSkill);

	PlayerInputComponent->BindTouch(IE_Pressed, this, &ATP_SideScrollerCharacter::TouchStarted);
	PlayerInputComponent->BindTouch(IE_Released, this, &ATP_SideScrollerCharacter::TouchStopped);
}


//---------- Move ----------


void ATP_SideScrollerCharacter::myJump() {

	if (!isAttack) {
		bPressedJump = true;
		JumpKeyHoldTime = 0.0f;
	}

}

void ATP_SideScrollerCharacter::MoveRight(float Value)
{
	if (canMove && !isAttack) {
		//face front when backstep
		GetCharacterMovement()->bOrientRotationToMovement = true;
		// add movement in that direction
		AddMovementInput(FVector(0.f, 0.5f, 0.f), Value);
	}

}

void ATP_SideScrollerCharacter::MoveLeft(float Value)
{
	if (canMove && !isAttack) {
		//do not face front when backstep
		GetCharacterMovement()->bOrientRotationToMovement = false;
		// add movement in that direction
		AddMovementInput(FVector(0.f, -0.5f, 0.f), Value);
	}
}


void ATP_SideScrollerCharacter::moveCrouch() {

	UE_LOG(LogTemp, Log, TEXT("This is moveCrouch"));
	wasCrouchUsed = true;
	canMove = false;
}
void ATP_SideScrollerCharacter::stopCrouch() {
	UE_LOG(LogTemp, Log, TEXT("This is stopCrouch"));
	wasCrouchUsed = false;
	canMove = true;
}

//---------- Avoid ----------

void ATP_SideScrollerCharacter::avoidBlock() {
	UE_LOG(LogTemp, Log, TEXT("This is avoidBlock"));
	wasBlockUsed = true;
	canMove = false;
}
void ATP_SideScrollerCharacter::stopBlock() {
	UE_LOG(LogTemp, Log, TEXT("This is stopBlock"));
	wasBlockUsed = false;
	canMove = true;
}


void ATP_SideScrollerCharacter::animHurt() {
	UE_LOG(LogTemp, Log, TEXT("This is HurtAnim"));
	wasHurtUsed = false;
	isAttack = false;
}

void ATP_SideScrollerCharacter::animHurtCombo() {
	UE_LOG(LogTemp, Log, TEXT("This is animHurtCombo"));
	wasHurtComboUsed = false;
	isAttack = false;
}

void ATP_SideScrollerCharacter::animDead() {
	UE_LOG(LogTemp, Log, TEXT("This is DeadAnim"));
	wasDeadUsed = false;
	isAttack = false;
}



//---------- Attack ----------

void ATP_SideScrollerCharacter::attackPunch_RPress() {
	canUsePunchCombo = true;
	canUsePunchKickCombo = true;
}

void ATP_SideScrollerCharacter::attackKickPress() {
	canUseKickCombo = true;
}

void ATP_SideScrollerCharacter::attackPunch() {

	if (canUsePunchCombo) {
		UE_LOG(LogTemp, Log, TEXT("This is attackPunchCombo"));
		wasPunchComboUsed = true;
		canUsePunchCombo = false;
	}
	else {
		UE_LOG(LogTemp, Log, TEXT("This is attackPunch"));
		wasPunchAttackUsed = true;
	}
}

void ATP_SideScrollerCharacter::attackPunch_R() {

	UE_LOG(LogTemp, Log, TEXT("This is attackPunch_R"));
	wasPunch_R_AttackUsed = true;
	canUsePunchCombo = false;
	canUsePunchKickCombo = false;;

}


void ATP_SideScrollerCharacter::attackKick_L() {
	if (canUseKickCombo) {
		UE_LOG(LogTemp, Log, TEXT("This is KickCombo"));
		wasKickComboUsed = true;
		canUseKickCombo = false;
	}
	else if (canUsePunchKickCombo) {
		UE_LOG(LogTemp, Log, TEXT("This is KickPunchCombo"));
		wasPunchKickComboUsed = true;
		canUsePunchKickCombo = false;
	}
	else {
		UE_LOG(LogTemp, Log, TEXT("This is attackKicK_L"));
		wasKick_L_AttackUsed = true;
	}

}

void ATP_SideScrollerCharacter::attackKick() {
	UE_LOG(LogTemp, Log, TEXT("This is attackKicK"));
	wasKickAttackUsed = true;
	canUseKickCombo = false;
}

void ATP_SideScrollerCharacter::attackFar() {

	UE_LOG(LogTemp, Log, TEXT("This is FarAttack"));
	wasFarAttackUsed = true;
	//canMove = false;
};

void ATP_SideScrollerCharacter::attackSkill() {

	if (playerStamina >= 1) {
		UE_LOG(LogTemp, Log, TEXT("This is attackSkill"));
		wasSkillAttackUsed = true;
		playerStamina = 0.00f;
		//canMove = false;
	}
	else {
		UE_LOG(LogTemp, Log, TEXT("Need more Stamina"));
	}

}

void ATP_SideScrollerCharacter::TouchStarted(const ETouchIndex::Type FingerIndex, const FVector Location)
{
	// jump on any touch
	Jump();
}

void ATP_SideScrollerCharacter::TouchStopped(const ETouchIndex::Type FingerIndex, const FVector Location)
{
	StopJumping();
}

//---------- Health ----------

void ATP_SideScrollerCharacter::TakeDamage(float _damageAmount) {
	UE_LOG(LogTemp, Log, TEXT("We are taking damage for %f points"), _damageAmount);
	playerHealth -= _damageAmount;
	if (playerHealth < 0.00f) {
		playerHealth = 0.00f;
	}
}

void ATP_SideScrollerCharacter::TakeStamina(float _staminaAmount) {
	UE_LOG(LogTemp, Log, TEXT("We are taking stamina damage for %f points"), _staminaAmount);
	playerStamina += _staminaAmount;
	if (playerStamina < 0.00f) {
		playerStamina = 0.00f;
	}
}