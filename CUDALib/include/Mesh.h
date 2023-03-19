#pragma once

#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
using namespace std;
namespace JH
{
	class Mesh
	{
	public:

		Mesh(vector<unsigned int> attributeSizes, vector<float> packedVertices, vector<unsigned int> indices = vector<unsigned int>())
		{
			unsigned int stride = 0;
			for (int i = 0; i < attributeSizes.size(); i++)
			{
				stride += attributeSizes[i];
			}
			unsigned int numVertices = (unsigned int)packedVertices.size() / stride;

			for (unsigned int i = 0; i < numVertices; i++)
			{
				unsigned int baseV = stride * i;
				unsigned int baseN = (stride >= 6) ? baseV + 3 : baseV;
				unsigned int baseT = baseN + 3;

				m_positions.push_back(glm::vec3(packedVertices[baseV + 0], packedVertices[baseV + 1], packedVertices[baseV + 2]));
				if (stride >= 6)
				{
					m_normals.push_back(glm::vec3(packedVertices[baseN + 0], packedVertices[baseN + 1], packedVertices[baseN + 2]));
				}
				m_texCoords.push_back(glm::vec2(packedVertices[baseT + 0], packedVertices[baseT + 1]));
			}
			Initialize(m_positions, m_normals, m_texCoords, indices, attributeSizes);
		}

		Mesh(const vector<glm::vec3>& vertices, const vector<glm::vec3>& normals = vector<glm::vec3>(),
			const vector<glm::vec2>& texCoords = vector<glm::vec2>(), const vector<unsigned int>& indices = vector<unsigned int>())
		{
			Initialize(vertices, normals, texCoords, indices);
		}

		Mesh(const Mesh&) = delete;

		bool useIndices() const
		{
			return m_indices.size() > 0;
		}

		unsigned int drawCount() const
		{
			if (useIndices())
			{
				return (unsigned int)m_indices.size();
			}
			else
			{
				return (unsigned int)m_positions.size();
			}
		}

		const vector<glm::vec3>& vertices() const
		{
			return m_positions;
		}

		const vector<unsigned int>& indices() const
		{
			return m_indices;
		}

		const vector<glm::vec3>& normals() const
		{
			return m_normals;
		}
		const vector<glm::vec2>& uvs() const
		{
			return m_texCoords;
		}
		void SetVerticesAndNormals(const vector<glm::vec3>& vertices, const vector<glm::vec3>& normals)
		{
			auto size = vertices.size() * sizeof(glm::vec3);
			m_positions = vertices;
			m_normals = normals;

		}


	private:
		vector<glm::vec3> m_positions;
		vector<glm::vec3> m_normals;
		vector<glm::vec2> m_texCoords;
		vector<unsigned int> m_indices;


		void Initialize(const vector<glm::vec3>& vertices, const vector<glm::vec3>& normals, const vector<glm::vec2>& texCoords,
			const vector<unsigned int>& indices, vector<unsigned int> attributeSizes = {})
		{
			m_positions = vertices;
			m_normals = normals;
			m_texCoords = texCoords;
			m_indices = indices;

		}

	};
}
