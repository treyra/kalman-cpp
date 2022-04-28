/**
* Implementation of KalmanFilter functions.
* Adapted from:
* @author: Hayk Martirosyan
* @date: 2014.11.15
*
* Updated to state transition matrix and input format and to make function based
* @author: Jimmy Ragan
* @date : 2022.4.27
*
*/

#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    double dt,
    const Eigen::MatrixXd& F,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P)
  : F(F), B(B), C(C), Q(Q), R(R), P0(P),
    m(C.rows()), n(F.rows()), dt(dt), initialized(false),
    I(n, n), x_hat(n), x_hat_new(n)
{
  I.setIdentity();
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
  x_hat = x0;
  P = P0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init() {
  x_hat.setZero();
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

// TODO: Make overload with u=0?
void KalmanFilter::update(const Eigen::VectorXd& y, const Eigen::VectorXd& u) {

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");

  x_hat_new = F * x_hat + B * u;
  P = F*P*F.transpose() + Q;
  K = P*C.transpose()*(C*P*C.transpose() + R).inverse();
  x_hat_new += K * (y - C*x_hat_new);
  P = (I - K*C)*P;
  x_hat = x_hat_new;

  t += dt;
}

void KalmanFilter::update(const Eigen::VectorXd& y, const Eigen::VectorXd& u, double dt, const Eigen::MatrixXd A) {

  this->A = A;
  this->dt = dt;
  update(y,u);
}
