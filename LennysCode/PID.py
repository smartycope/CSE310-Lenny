class PID:
    """
    Discrete PID control -- proportional- integral-derivative controller
    """
    def __init__(self, P=1.3, I=0.3, D=0.0, Derivator=0, Integrator=0, Integrator_max=500, Integrator_min=-500, model='builtin'):
        self.Kp=P
        self.Ki=I
        self.Kd=D
        self.Derivator=Derivator
        self.Integrator=Integrator
        self.Integrator_max=Integrator_max
        self.Integrator_min=Integrator_min
        self.set_point=0.0
        self.error=0.0

    def update(self, current_value): # Current value passed in is CFangleX (or CFangleX1)
        """
        Calculate PID output value for given reference input and feedback
        """
        if model == 'builtin':
            self.error = self.set_point - current_value
            self.P_value = self.Kp * self.error
            self.D_value = self.Kd * ( self.error - self.Derivator)
            self.Derivator = self.error
            self.Integrator = self.Integrator + self.error

            if self.Integrator > self.Integrator_max:
                self.Integrator = self.Integrator_max
            elif self.Integrator < self.Integrator_min:
                self.Integrator = self.Integrator_min

            self.I_value = self.Integrator * self.Ki

            return self.P_value + self.I_value + self.D_value
        elif model == 'integrator':
            pass
        else:
            raise TypeError(f'Unknown model {model}')

    def setPoint(self,set_point):
        """
        Initilize the setpoint of PID
        """
        self.set_point = set_point
        self.Integrator=0
        self.Derivator=0
