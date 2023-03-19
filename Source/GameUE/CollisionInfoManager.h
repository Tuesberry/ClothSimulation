// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Components/ActorComponent.h"
#include "CollisionInfo.h"
#include "PhysicsEngine/BodySetup.h"
#include "DrawDebugHelpers.h"
#include "CollisionInfoManager.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class GAMEUE_API UCollisionInfoManager : public UActorComponent
{
	GENERATED_BODY()
public:
	// Sets default values for this component's properties
	UCollisionInfoManager();

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

	bool m_initSamplingInfo = false;
public:
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(BlueprintReadOnly)
		ACharacter* m_ownerActor;

	UPROPERTY(BlueprintReadOnly)
		UPhysicsAsset* m_physicsAsset;

	/** Physics asset to use. If empty use the skeletal mesh's default physics asset */
	UPROPERTY(EditAnywhere, Category = Settings)
		UPhysicsAsset* m_overridePhysicsAsset = nullptr;

	UPROPERTY(BlueprintReadOnly)
		TArray<FCollisionInfo> m_collisionArr;

	UPROPERTY(BlueprintReadOnly)
		TArray<FCollisionInfo> m_BeforeCollisionArr;

	UFUNCTION(BlueprintCallable)
		void CollisionSampling();

	UFUNCTION(BlueprintCallable)
		void DebugCurCollision(float deltaTime);

	UFUNCTION(BlueprintCallable)
		void DebugBeforeCollision(float deltaTime);
};
