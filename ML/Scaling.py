import numpy as np
import logging


class Scaler:
    logger = logging.getLogger(__name__)

    def __init__(self, args):
        self.args = args

    def scaleXy(self, X, y):
        if y is not None:
            if self.args.scale_y: 
                xHI = y[:, 0].reshape(len(y), 1)
                scaledfx = (0.8 + y[:,1]/5.0).reshape(len(y), 1)
                y = np.hstack((xHI, scaledfx))
            if self.args.scale_y0: y[:,0] = y[:,0]*5.0
            if self.args.scale_y1:
                # we wish to create a single metric representing the expected
                # strength of the signal based on xHI (0 to 1) and logfx (-4 to +1)
                # We know that higher xHI and lower logfx lead to a stronger signal, 
                # First we scale logfx to range of 0 to 1.
                # Then we take a product of xHI and (1 - logfx)
                if self.args.trials == 1: logger.info(f"Before scaleXy: {y}")
                xHI = y[:, 0].reshape(len(y), 1)
                scaledfx = 1 - (0.8 + y[:,1]/5.0)
                product = np.sqrt(xHI**2 + scaledfx**2).reshape(len(y), 1)
                y = np.hstack((xHI, scaledfx, product))
                if self.args.trials == 1: logger.info(f"ScaledXy: {y}")
            if self.args.scale_y2:
                # we wish to create a single metric representing the expected
                # strength of the signal based on xHI (0 to 1) and logfx (-4 to +1)
                # We know that higher xHI and lower logfx lead to a stronger signal, 
                # First we scale logfx to range of 0 to 1.
                # Then we take a product of xHI and (1 - logfx)
                if self.args.trials == 1: logger.info(f"Before scaleXy: {y}")
                xHI = y[:, 0].reshape(len(y), 1)
                scaledfx = 1 - (0.8 + y[:,1]/5.0).reshape(len(y), 1)
                product = np.sqrt(xHI**2 + scaledfx**2).reshape(len(y), 1)
                y = np.hstack((xHI, scaledfx, product))
                if self.args.trials == 1: logger.info(f"ScaledXy: {y}")
        if X is not None:
            print(f"### Scaler: Logscale X applied")
            if self.args.logscale_X: X = np.log(np.clip(X, 1e-20, None))
        return X, y

    def unscaleXy(self, X, y):
        # Undo what we did in scaleXy function
        if self.args.scale_y: 
            xHI = y[:, 0].reshape(len(y), 1)
            fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
            y = np.hstack((xHI, fx))
        elif self.args.scale_y0: y[:,0] = y[:,0]/5.0
        elif self.args.scale_y1:
            xHI = y[:, 0].reshape(len(y), 1)
            fx = 5.0*(1 - y[:,1] - 0.8)
            y = np.hstack((xHI, fx))
        elif self.args.scale_y2:
            xHI = y[:, 0].reshape(len(y), 1)
            fx = 5.0*(1 - y[:,1] - 0.8).reshape(len(y), 1)
            y = np.hstack((xHI, fx))
                    
        if X is not None:
            if self.args.logscale_X: X = np.exp(X)
        return X, y

    def unscale_y(self, y):
        # Undo what we did in the scaleXy function
        if self.args.scale_y: 
            xHI = y[:, 0].reshape(len(y), 1)
            fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
            y = np.hstack((xHI, fx))
        elif self.args.scale_y0: y[:,0] = y[:,0]/5.0
        elif self.args.scale_y1:
            # calculate fx using product and xHI 
            if self.args.trials == 1: logger.info(f"Before unscale_y: {y}")
            xHI = np.sqrt(y[:,2]**2 - y[:,1]**2)
            fx = 5.0*(1 - y[:,1] - 0.8)
            if self.args.trials == 1: logger.info(f"Unscaled_y: {y}")
            y = np.hstack((xHI, fx))
        elif self.args.scale_y2:
            # calculate fx using product and xHI 
            if self.args.trials == 1: logger.info(f"Before unscale_y: {y}")
            xHI = np.sqrt(0.5*y**2).reshape((len(y), 1))
            fx = 5.0*(1 - xHI - 0.8)
            y = np.hstack((xHI, fx))
            if self.args.trials == 1: logger.info(f"Unscaled_y: {y}")
        return y