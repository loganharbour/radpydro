import numpy as np
import matplotlib.pyplot as plt

class Fields:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.N = rp.geo.N

        # Velocities (spatial cell edges)
        self.u = self.initializeAtEdges(self.input.u)
        self.u_p = np.copy(self.u)
        self.u_old = np.copy(self.u)
        self.u_IC = np.copy(self.u)

        # Set left velocity BCs as necessary
        if self.input.hydro_L == 'u':
            if self.input.hydro_L_val == None:
                self.u_L = self.input.u(0)
            else:
                self.u_L = self.input.hydro_L_val
                self.u_IC[0] = self.u_L;
                self.u_old[0] = self.u_L;
                self.u_p[0] = self.u_L;
                self.u[0] = self.u_L;
        else:
            self.u_L = None

        # Set right velocity BCs as necessary
        if self.input.hydro_R == 'u':
            if self.input.hydro_R_val == None:
                self.u_R = self.input.u(self.geo.r_half_old[-1])
            else:
                self.u_R = self.input.hydro_R_val
                self.u_IC[-1] = self.u_R;
                self.u_old[-1] = self.u_R;
                self.u_p[-1] = self.u_R;
                self.u[-1] = self.u_R;
        else:
            self.u_R = None

        # Temperature (spatial cell centers)
        self.T = self.initializeAtCenters(self.input.T)
        self.T_p = np.copy(self.T)
        self.T_old = np.copy(self.T)

        # Densities (spatial cell centers)
        self.rho = self.initializeAtCenters(self.input.rho)
        self.rho_p = np.copy(self.rho)
        self.rho_old = np.copy(self.rho)
        self.rho_IC = np.copy(self.rho)

        # Pressures (spatial cell centers, IC: Eqs. 22 and 23)
        self.P = (self.mat.gamma - 1) * self.mat.C_v * self.T * self.rho
        self.P_p = np.copy(self.P)
        self.P_old = np.copy(self.P)

        # Defining artifical viscosity
        self.Q = np.zeros(self.N)

        # Set pressure BCs as necessary
        if self.input.hydro_L == 'P':
            self.P_L = self.input.hydro_L_val
        else:
            self.P_L = None
        if self.input.hydro_R == 'P':
            self.P_R = self.input.hydro_R_val
        else:
            self.P_R = None

        # Internal energies (spatial cell centers)
        self.e = self.mat.C_v * self.T_old
        self.e_p = np.copy(self.e)
        self.e_old = np.copy(self.e)
        self.e_IC = np.copy(self.e)

        # Radiation energies (spatial cell centers)
        self.E = self.initializeAtCenters(self.input.E)
        self.E_p = np.copy(self.E)
        self.E_old = np.copy(self.E)
        self.E_IC = np.copy(self.E)

        # Set E boundary conditions
        if self.input.rad_L == 'source':
            self.E_bL = self.input.rad_L_val
        else:
            self.E_bL = None
        if self.input.rad_R == 'source':
            self.E_bR = self.input.rad_R_val
        else:
            self.E_bR = None

        # Init the rest of the materials that depend on field variables
        self.mat.initFromFields(self)

    # Copy over all new fields to old positions
    def stepFields(self):
        np.copyto(self.u_old, self.u)
        np.copyto(self.T_old, self.T)
        np.copyto(self.rho_old, self.rho)
        np.copyto(self.P_old, self.P)
        np.copyto(self.e_old, self.e)
        np.copyto(self.E_old, self.E)

    # Initialize variable with function at the spatial cell centers
    def initializeAtCenters(self, function):
        values = np.zeros(self.N)
        if function is not None:
            for i in range(self.N):
                values[i] = function(self.geo.r[i])
        return values

    # Initialize variable with function at the spatial cell edges
    def initializeAtEdges(self, function):
        values = np.zeros(self.N + 1)
        if function is not None:
            for i in range(self.N + 1):
                values[i] = function(self.geo.r_half[i])
        return values

    def addArtificialViscosity(self):
        # Initializing references for shorter notations
        rho = self.rho_old
        gamma = self.mat.gamma
        u = self.u_old
        dr = self.geo.dr_old
        P = self.P_old

        # Looping over cells to compute the cell-wise viscosity
        for i in range(self.N):
            du = u[i+1] - u[i]

            if (du >= 0):
                self.Q[i] = 0
                continue

            # Left boundary cell
            if (i==0):
                rho_minus = rho[i]
                rho_plus = (rho[i] * dr[i] + rho[i+1] * dr[i+1]) / (dr[i] + dr[i+1])

                c_s_minus = (gamma * P[i] / rho[i])**(1 / 2)
                c_s_plus = ((gamma * P[i] / rho[i])**(1 / 2) * dr[i] +  \
                            (gamma * P[i+1] / rho[i+1])**(1 / 2) * dr[i+1]) \
                            / (dr[i] + dr[i+1])

                R_minus = 1
                R_plus = ((u[i+2] - u[i+1]) * dr[i]) / ((u[i+1] - u[i]) * dr[i+1])

            # Internal cells
            elif (i > 0 and i < self.N - 1):

                rho_minus = (rho[i-1] * dr[i-1] + rho[i] * dr[i]) / (dr[i-1] + dr[i])
                rho_plus = (rho[i] * dr[i] + rho[i+1] * dr[i+1]) / (dr[i] + dr[i+1])

                c_s_minus = ((gamma * P[i-1] / rho[i-1])**(1 / 2) * dr[i-1] +  \
                             (gamma * P[i] / rho[i])**(1 / 2) * dr[i]) \
                            / (dr[i-1] + dr[i])
                c_s_plus = ((gamma * P[i] / rho[i])**(1 / 2) * dr[i] +  \
                            (gamma * P[i+1] / rho[i+1])**(1 / 2) * dr[i+1]) \
                            / (dr[i] + dr[i+1])

                R_minus = ((u[i] - u[i-1]) * dr[i]) / ((u[i+1] - u[i]) * dr[i-1])
                R_plus = ((u[i+2] - u[i+1]) * dr[i]) / ((u[i+1] - u[i]) * dr[i+1])

            # Right boundary cell
            elif (i == self.N - 1):
                rho_minus = (rho[i-1] * dr[i-1] + rho[i] * dr[i]) / (dr[i-1] + dr[i])
                rho_plus = rho[i]

                c_s_minus = ((gamma * P[i-1] / rho[i-1])**(1 / 2) * dr[i-1] +  \
                             (gamma * P[i] / rho[i])**(1 / 2) * dr[i]) \
                            / (dr[i-1] + dr[i])
                c_s_plus = (gamma * P[i] / rho[i])**(1 / 2)

                R_minus = ((u[i] - u[i-1]) * dr[i]) / ((u[i+1] - u[i]) * dr[i-1])
                R_plus = 1

            # Computing artificial viscosity
            rho_bar = 2 * (rho_minus * rho_plus) / (rho_minus + rho_plus)
            c_s_bar = min(c_s_plus, c_s_minus)
            c_Q = (gamma + 1) / 4

            T = max(0, min(1, 2 * R_minus, 2 * R_plus, 0.5 * (R_minus + R_plus)))

            self.Q[i] = (1 - T) * rho_bar * abs(du) * \
                        (c_Q * abs(du) + (c_Q**2 * du**2 + c_s_bar**2)**(1 / 2))

        # Updating pressure:
        self.P_old += self.Q


    def plotFields(self):
        fig, ax = plt.subplots(nrows=2, ncols=3)
        titles = [['Density', 'Velocity', 'Internal Enegy'],
                  ['Radiation Energy', 'Temperature', 'Pressure']]
        x_axis = [[self.geo.r, self.geo.r_half, self.geo.r],
                  [self.geo.r, self.geo.r,      self.geo.r]]
        y_axis = [[self.rho, self.u, self.e],
                  [self.E,   self.T, self.P]]


        for i in range(2):
            for j in range(3):
                for k in range(len(y_axis[i][j])):
                    y_axis[i][j][k] = np.round(y_axis[i][j][k], 5)
                ax[i][j].plot(x_axis[i][j], y_axis[i][j])
                ax[i][j].set_title(titles[i][j])
                ax[i][j].set_xlim([-0.02, 0.02])
        plt.tight_layout()
        plt.show()
