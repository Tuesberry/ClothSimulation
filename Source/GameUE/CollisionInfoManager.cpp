// Fill out your copyright notice in the Description page of Project Settings.

#include "CollisionInfoManager.h"

// Sets default values for this component's properties
UCollisionInfoManager::UCollisionInfoManager()
	: m_ownerActor(nullptr)
	, m_physicsAsset(nullptr)
	, m_overridePhysicsAsset(nullptr)
	, m_collisionArr()
	, m_BeforeCollisionArr()
	, m_initSamplingInfo(false)
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = true;

	// ...
}


// Called when the game starts
void UCollisionInfoManager::BeginPlay()
{
	Super::BeginPlay();

	// get owner character
	m_ownerActor = Cast<ACharacter>(GetOwner());

	if (m_ownerActor == nullptr)
	{
		UE_LOG(LogTemp, Log, TEXT("Owned actor is not a character"));
	}
	else
	{
		const USkeletalMeshComponent* skeletalMeshComp = m_ownerActor->GetMesh()->GetAnimInstance()->GetSkelMeshComponent();
		const USkeletalMesh* skeletalMeshAsset = skeletalMeshComp->SkeletalMesh;

		const FReferenceSkeleton& skelMeshRefSkel = skeletalMeshAsset->RefSkeleton;
		m_physicsAsset = m_overridePhysicsAsset ? m_overridePhysicsAsset : m_ownerActor->GetMesh()->GetPhysicsAsset();

		// collision Sampling 
		CollisionSampling();
	}
}

void UCollisionInfoManager::CollisionSampling()
{
	// get before collision info
	m_BeforeCollisionArr = m_collisionArr;

	const USkeletalMeshComponent* skeletalMeshComp = m_ownerActor->GetMesh()->GetAnimInstance()->GetSkelMeshComponent();
	int arrIdx = 0;
	FVector actorWorldLoc = m_ownerActor->GetActorLocation();
	// get info from physics assets
	for (int32 i = 0; i < m_physicsAsset->SkeletalBodySetups.Num(); i++)
	{
		if (!ensure(m_physicsAsset->SkeletalBodySetups[i]))
		{
			continue;
		}
		int32 boneIdx = skeletalMeshComp->GetBoneIndex(m_physicsAsset->SkeletalBodySetups[i]->BoneName);

		// if we found a bone for it, add it to the array
		if (boneIdx != INDEX_NONE)
		{
			FTransform boneTM = skeletalMeshComp->GetBoneTransform(boneIdx);
			float scale = boneTM.GetScale3D().GetAbsMax();
			FVector vectorScale(scale);
			boneTM.RemoveScaling();

			FKAggregateGeom& AggGeom = m_physicsAsset->SkeletalBodySetups[i]->AggGeom;

			for (int j = 0; j < AggGeom.SphereElems.Num(); j++)
			{
				FTransform elemTM = AggGeom.SphereElems[j].GetTransform();
				elemTM.ScaleTranslation(vectorScale);
				elemTM *= boneTM;

				FCollisionInfo newInfo;
				newInfo.m_collisionType = ECollisionType::Sphere;
				newInfo.m_radius = AggGeom.SphereElems[j].Radius * scale;
				newInfo.m_height = 0;
				newInfo.m_extent = FVector();
				newInfo.m_position = elemTM.GetLocation();
				newInfo.m_rotation = elemTM.GetRotation();

				if (m_initSamplingInfo)
				{
					m_collisionArr[arrIdx] = newInfo;
				}
				else
				{
					m_collisionArr.Add(newInfo);
				}

				arrIdx++;
			}

			for (int j = 0; j < AggGeom.BoxElems.Num(); j++)
			{
				FTransform elemTM = AggGeom.BoxElems[j].GetTransform();
				elemTM.ScaleTranslation(vectorScale);
				elemTM *= boneTM;

				FCollisionInfo newInfo;
				newInfo.m_collisionType = ECollisionType::Box;
				newInfo.m_radius = 0;
				newInfo.m_height = 0;
				newInfo.m_position = elemTM.GetLocation();
				newInfo.m_extent = scale * 0.5f * FVector(AggGeom.BoxElems[j].X, AggGeom.BoxElems[j].Y, AggGeom.BoxElems[j].Z);
				newInfo.m_rotation = elemTM.GetRotation();
				if (m_initSamplingInfo)
				{
					m_collisionArr[arrIdx] = newInfo;
				}
				else
				{
					m_collisionArr.Add(newInfo);
				}

				arrIdx++;
			}

			for (int j = 0; j < AggGeom.SphylElems.Num(); j++)
			{
				FTransform elemTM = AggGeom.SphylElems[j].GetTransform();
				elemTM.ScaleTranslation(vectorScale);
				elemTM *= boneTM;

				FCollisionInfo newInfo;
				newInfo.m_collisionType = ECollisionType::Sphyl;
				newInfo.m_radius = AggGeom.SphylElems[j].Radius * scale;
				newInfo.m_height = AggGeom.SphylElems[j].Length * scale;
				newInfo.m_position = elemTM.GetLocation();
				newInfo.m_extent = FVector();
				newInfo.m_rotation = elemTM.GetRotation();

				if (m_initSamplingInfo)
				{
					m_collisionArr[arrIdx] = newInfo;
				}
				else
				{
					m_collisionArr.Add(newInfo);
				}

				arrIdx++;
			}
		}
	}
	if (!m_initSamplingInfo)
	{
		m_initSamplingInfo = true;
	}
}
// Called every frame
void UCollisionInfoManager::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	CollisionSampling();
	// ...
}

void UCollisionInfoManager::DebugCurCollision(float deltaTime)
{
	for (FCollisionInfo& info : m_collisionArr)
	{
		FVector actorWorldLoc = m_ownerActor->GetActorLocation();

		UWorld* world = GetWorld();
		if (world)
		{
			if (info.m_collisionType == ECollisionType::Sphere)
			{
				DrawDebugSphere(world, info.m_position + actorWorldLoc, info.m_radius, 16, FColor(181, 0, 0), false, deltaTime, 0, 1);
			}
			else if (info.m_collisionType == ECollisionType::Box)
			{
				DrawDebugBox(world, info.m_position + actorWorldLoc, info.m_extent, info.m_rotation, FColor(181, 0, 0), false, deltaTime, 0, 1);
			}
			else if (info.m_collisionType == ECollisionType::Sphyl)
			{
				DrawDebugCapsule(world, info.m_position + actorWorldLoc, info.m_height, info.m_radius, info.m_rotation, FColor(181, 0, 0), false, deltaTime, 0, 1);
			}
		}
	}
}

void UCollisionInfoManager::DebugBeforeCollision(float deltaTime)
{
	for (FCollisionInfo& info : m_BeforeCollisionArr)
	{
		FVector actorWorldLoc = m_ownerActor->GetActorLocation();

		UWorld* world = GetWorld();
		if (world)
		{
			if (info.m_collisionType == ECollisionType::Sphere)
			{
				DrawDebugSphere(world, info.m_position + actorWorldLoc, info.m_radius, 16, FColor(0, 181, 0), false, deltaTime, 0, 1);
			}
			else if (info.m_collisionType == ECollisionType::Box)
			{
				DrawDebugBox(world, info.m_position + actorWorldLoc, info.m_extent, info.m_rotation, FColor(0, 181, 0), false, deltaTime, 0, 1);
			}
			else if (info.m_collisionType == ECollisionType::Sphyl)
			{
				DrawDebugCapsule(world, info.m_position + actorWorldLoc, info.m_height, info.m_radius, info.m_rotation, FColor(0, 181, 0), false, deltaTime, 0, 1);
			}
		}
	}
}