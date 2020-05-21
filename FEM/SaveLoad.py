import pickle as pk
import gmsh

class SaveLoad():

    def save(self,name,saved_ddls,ddl,sln,sln_main,tspan):
        """Save the mesh and the main infos about the numerical result."""
        self.ddl = ddl
        self.sln = sln
        self.sln_main = sln_main
        self.tspan = tspan
        self.saved_ddls = saved_ddls
        gmsh.option.setNumber('Mesh.SaveAll',1)
        gmsh.write('data/' + name + '.msh')
        with open('data/' + name + '.pickle', 'wb') as f:
            pk.dump(self, f, pk.HIGHEST_PROTOCOL)

    def load(self,name,m):
        """Load the mesh and the main infos about a numerical result."""
        gmsh.open('data/' + name + '.msh')
        m.readMsh('data/' + name + '.msh')
        with open('data/' + name + '.pickle', 'rb') as f:
            T = pk.load(f)
            self.ddl = T.ddl
            self.sln = T.sln
            self.sln_main = T.sln_main
            self.tspan = T.tspan
            self.saved_ddls = T.saved_ddls