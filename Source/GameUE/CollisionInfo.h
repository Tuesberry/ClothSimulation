// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CollisionInfo.generated.h"

UENUM(BlueprintType)
enum class ECollisionType : uint8
{
	Sphere UMETA(DisplayName = "Sphere"),
	Box UMETA(DisplayName = "Box"),
	Sphyl UMETA(DisplayName = "Sphyl")
};

USTRUCT(BlueprintType)
struct FCollisionInfo
{
public:
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CollisionInfo")
	ECollisionType m_collisionType;

	// radius
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CollisionInfo")
	float m_radius;

	// height
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CollisionInfo")
	float m_height;

	// world position
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CollisionInfo")
	FVector m_position;

	// used for box collision
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CollisionInfo")
	FVector m_extent;	

	// rotation(quaternion)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "CollisionInfo")
	FQuat m_rotation;
};
