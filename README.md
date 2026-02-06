# Nelson-Siegel
This project implements the Diebold-Li dynamic formulation of the Nelson-Siegel factor model for fitting the term structure of interest rates.
It is based on the seminal paper "Forecasting the Term Structure of Government Bond Yields" (Diebold & Li, 2006).

## The Diebold-Li Framework
The model fits the following equation at each time $t$:

$$y_t(\tau) = \beta_{1,t} + \beta_{2,t} \left( \frac{1 - e^{-\lambda \tau}}{\lambda \tau} \right) + \beta_{3,t} \left( \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau} \right)$$

Unlike the original 1987 nonlinear approach, Diebold-Li fixes $\lambda$ (or estimates it globally) to make the system linear. This code implements that specific methodology:
* **Linearization**: The factor loadings (terms in brackets) are pre-calculated based on $\lambda$.
* **OLS Estimation**: The factors $\beta_{1,t}, \beta_{2,t}, \beta_{3,t}$ are estimated using Ordinary Least Squares for every time step.
* **Default Lambda**: The code defaults to $\lambda = 0.0609$, the value proposed by Diebold-Li to maximize curvature loading at 30 months (2.5 years).

## Outputs
The script produces the **Fitted Yield Curve** and two dashboards:
* Factor Dynamics
* Fit Diagnostics

## References
* Diebold, F. X., & Li, C. (2006). _Forecasting the Term Structure of Government Bond Yields._ Journal of Econometrics, 130(2), 337-364.
* Nelson, C. R., & Siegel, A. F. (1987). _Parsimonious Modeling of Yield Curves._  Journal of Business, 473-489.
