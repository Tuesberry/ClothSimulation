// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class GameUE : ModuleRules
{
	private string project_root_path
	{
		get { return Path.Combine(ModuleDirectory, "../.."); }
	}
	public GameUE(ReadOnlyTargetRules Target) : base(Target)
	{

        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay" });

        PrivateDependencyModuleNames.AddRange(new string[] { "ProceduralMeshComponent" });


        string custom_cuda_lib_include = "CUDALib/include";
        string custom_cuda_lib_lib = "CUDALib/lib";

        PublicIncludePaths.Add(Path.Combine(project_root_path, custom_cuda_lib_include));
        PublicAdditionalLibraries.Add(Path.Combine(project_root_path, custom_cuda_lib_lib, "ClothSimulationCUDA.lib"));

        string cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7";
        string cuda_include = "include";
        string cuda_lib = "lib/x64";

        PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));

        //PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));

    }
}
