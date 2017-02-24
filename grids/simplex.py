# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:31:31 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps
import scipy.spatial

from core.grids.grid import Grid
from utils import setmembership
from utils import accumarray


class TriangleGrid(Grid):

    def __init__(self, p, tri=None):
        """
        Create triangular grid from point cloud.

        If no triangulation is provided, Delaunay will be applied.

        Examples:
        >>> p = np.random.rand(2, 10)
        >>> tri = scipy.spatial.Delaunay(p.transpose()).simplices
        >>> g = TriangleGrid(p, tri.transpose())

        Parameters
        ----------
        p (np.ndarray, 2 x num_nodes): Point coordinates
        tri (np.ndarray, 3 x num_cells): Cell-node connections. If not
        provided, a Delaunay triangulation will be applied
        """

        self.dim = 2

        # Transform points to column vector if necessary (scipy.Delaunay
        # requires this format)
        pdims = p.shape

        if p.shape[0] != 2:
            raise NotImplementedError("Have not yet implemented triangle grids "
                                      "embeded in 2D")
        if tri is None:
            tri = scipy.spatial.Delaunay(p.transpose())
            tri = tri.simplices
            tri = tri.transpose()

        num_nodes = p.shape[1]

        # Add a zero z-coordinate
        nodes = np.vstack( (p, np.zeros(num_nodes)) )
        assert num_nodes > 2   # Check of transposes of point array

        # Face node relations
        face_nodes = np.hstack((tri[[0, 1]],
                                tri[[1, 2]],
                                tri[[2, 0]])).transpose()
        face_nodes.sort(axis=1)
        face_nodes, tmp, cell_faces = setmembership.unique_rows(face_nodes)

        num_faces = face_nodes.shape[0]
        num_cells = tri.shape[1]

        num_nodes_per_face = 2
        face_nodes = face_nodes.ravel(0)
        indptr = np.hstack((np.arange(0, num_nodes_per_face * num_faces,
                                      num_nodes_per_face),
                            num_nodes_per_face * num_faces))
        data = np.ones(face_nodes.shape, dtype=bool)
        face_nodes = sps.csc_matrix((data, face_nodes, indptr),
                                    shape=(num_nodes, num_faces))

        # Cell face relation
        num_faces_per_cell = 3
        cell_faces = cell_faces.reshape(num_faces_per_cell, num_cells).ravel(1)
        indptr = np.hstack((np.arange(0, num_faces_per_cell*num_cells,
                                      num_faces_per_cell),
                            num_faces_per_cell * num_cells))
        data = -np.ones(cell_faces.shape)
        tmp, sgns = np.unique(cell_faces, return_index=True)
        data[sgns] = 1
        cell_faces = sps.csc_matrix((data, cell_faces, indptr),
                                    shape=(num_faces, num_cells))

        super(TriangleGrid, self).__init__(2, nodes, face_nodes, cell_faces,
                                           'TriangleGrid')

    def cell_node_matrix(self):
        """ Get cell-node relations in a Nc x 3 matrix
        Perhaps move this method to a superclass when tet-grids are implemented
        """

        # Absolute value needed since cellFaces can be negative
        cn = self.face_nodes * np.abs(self.cell_faces) \
             * sps.eye(self.num_cells)
        row, col = cn.nonzero()
        scol = np.argsort(col)

        # Consistency check
        assert np.all(accumarray.accum(col, np.ones(col.size)) ==
                      (self.dim + 1))

        return row[scol].reshape(self.num_cells, 3)


class StructuredTriangleGrid(TriangleGrid):

    def __init__(self, nx, physdims=None):
        """
        Construct a triangular grid by splitting Cartesian cells in two.

        Examples:
        Grid on the unit cube
        >>> nx = np.array([2, 3])
        >>> physdims = np.ones(2)
        >>> g = simplex.StructuredTriangleGrid(nx, physdims)

        Parameters
        ----------
        nx (np.ndarray, size 2): number of cells in each direction of
        underlying Cartesian grid
        physdims (np.ndarray, size 2): domain size. Defaults to nx,
        thus Cartesian cells are unit squares
        """
        nx = np.asarray(nx)
        if physdims is None:
            physdims = nx
        else:
            physdims = np.asarray(physdims)

        x = np.linspace(0, physdims[0], nx[0] + 1)
        y = np.linspace(0, physdims[1], nx[1] + 1)

        # Node coordinates
        x_coord, y_coord = np.meshgrid(x, y)
        p = np.vstack((x_coord.ravel(order='C'), y_coord.ravel(order='C')))

        # Define nodes of the first row of cells.
        tmp_ind = np.arange(0, nx[0])
        i1 = tmp_ind  # Lower left node in quad
        i2 = tmp_ind + 1  # Lower right node
        i3 = nx[0] + 2 + tmp_ind  # Upper left node
        i4 = nx[0] + 1 + tmp_ind  # Upper right node

        # The first triangle is defined by (i1, i2, i3), the next by
        # (i1, i3, i4). Stack these vertically, and reshape so that the
        # first quad is split into cells 0 and 1 and so on
        tri_base = np.vstack((i1, i2, i3, i1, i3, i4)).reshape((3, -1),
                                                               order='F')
        # Initialize array of triangles. For the moment, we will append the
        # cells here, but we do know how many cells there are in advance,
        # so pre-allocation is possible if this turns out to be a bottleneck
        tri = tri_base

        # Loop over all remaining rows in the y-direction.
        for iter1 in range(nx[1].astype('int64') - 1):
            # The node numbers are increased by nx[0] + 1 for each row
            tri = np.hstack((tri, tri_base + (iter1 + 1) * (nx[0] + 1)))

        super(self.__class__, self).__init__(p, tri)


class TetrahedralGrid(Grid):

    def __init__(self, p, tet=None):

        self.dim = 3

        # Transform points to column vector if necessary (scipy.Delaunay
        # requires this format)
        pdims = p.shape

        if tet is None:
            tet = scipy.spatial.Delaunay(p.transpose())
            tet = tet.simplices
            tet = tet.transpose()

        num_nodes = p.shape[1]

        nodes = p
        assert num_nodes > 3   # Check of transposes of point array

        tet = self.__permute_nodes(p, tet)

        # Face node relations
        face_nodes = np.hstack((tet[[1, 0, 2]],
                                tet[[0, 1, 3]],
                                tet[[2, 0, 3]],
                                tet[[1, 2, 3]])).transpose()
        sort_ind = np.squeeze(np.argsort(face_nodes, axis=1))
        face_nodes.sort(axis=1)
        face_nodes, tmp, cell_faces = setmembership.unique_rows(face_nodes)

        num_faces = face_nodes.shape[0]
        num_cells = tet.shape[1]

        num_nodes_per_face = 3
        face_nodes = face_nodes.ravel(0)
        indptr = np.hstack((np.arange(0, num_nodes_per_face * num_faces,
                                      num_nodes_per_face),
                            num_nodes_per_face * num_faces))
        data = np.ones(face_nodes.shape, dtype=bool)
        face_nodes = sps.csc_matrix((data, face_nodes, indptr),
                                    shape=(num_nodes, num_faces))

        # Cell face relation
        num_faces_per_cell = 4
        cell_faces = cell_faces.reshape(num_faces_per_cell, num_cells).ravel(1)
        indptr = np.hstack((np.arange(0, num_faces_per_cell*num_cells,
                                      num_faces_per_cell),
                            num_faces_per_cell * num_cells))
        data = np.ones(cell_faces.shape)
        sgn_change = np.where(np.any(np.diff(sort_ind) == 1, axis=1))[0]
        data[sgn_change] = -1
        cell_faces = sps.csc_matrix((data, cell_faces, indptr),
                                    shape=(num_faces, num_cells))

        super(TetrahedralGrid, self).__init__(3, nodes, face_nodes, cell_faces,
                                           'TetrahedralGrid')

    def __permute_nodes(self, p, t):
        v = self.__triple_product(p, t)
        permute = np.where(v > 0)[0]
        t[:2, permute] = t[1::-1, permute]
        v2 = self.__triple_product(p, t)
        return t

    def __triple_product(self, p, t):
        px = p[0]
        py = p[1]
        pz = p[2]

        x = px[t]
        y = py[t]
        z = pz[t]

        dx = x[1:] - x[0]
        dy = y[1:] - y[0]
        dz = z[1:] - z[0]

        cross_x = dy[0] * dz[1] - dy[1] * dz[0]
        cross_y = dz[0] * dx[1] - dz[1] * dx[0]
        cross_z = dx[0] * dy[1] - dx[1] * dy[0]

        return dx[2] * cross_x + dy[2] * cross_y + dz[2] * cross_z

