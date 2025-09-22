



class Face:

    def __init__(self, vertex_indices: list[int]):
        """
        A face in the mesh.

        :param vertex_indices: List of indices into the mesh's vertex list
        """
        # store the face vertices
        self.vertex_indices = vertex_indices
        # one halfedge bordering the face
        self.halfedge = None