import numpy as np
import mxnet.ndarray as mxn

def normalize(X1, X2, X3, X4):
    X1 = X1/mxn.sqrt(mxn.sum(mxn.square(X1), axis = 0) + (mxn.sum(mxn.square(X1), axis = 0)==0)*1e-20)
    X2 = X2/mxn.sqrt(mxn.sum(mxn.square(X2), axis = 0) + (mxn.sum(mxn.square(X2), axis = 0)==0)*1e-20)
    X3 = X3/mxn.sqrt(mxn.sum(mxn.square(X3), axis = 0) + (mxn.sum(mxn.square(X3), axis = 0)==0)*1e-20)
    X4 = X4/mxn.sqrt(mxn.sum(mxn.square(X4), axis = 0) + (mxn.sum(mxn.square(X4), axis = 0)==0)*1e-20)
    return X1, X2, X3, X4

def getn(X1, X2):
    N1 = X1
    dotp = X1*(mxn.sum(X1*X2, axis = 0))
    #dotp  = X1*np.einsum('ij,ij->j', X1, X2)
    N2 = X2 - dotp
    N2 = N2/mxn.sqrt(mxn.sum(N2*N2, axis = 0) + (mxn.sum(N2*N2, axis = 0)==0)*1e-20)
    #N2 = N2/np.sqrt(np.einsum('ij, ij->j', N2, N2))
    return N1, N2

def get_roots(A, B, C, D):
    alpha = (A*A - B*B + C*C - D*D)/(A*B + C*D+ (mxn.abs(A*B+C*D)<1e-10)*1e-10*(mxn.sign(A*B+C*D)) + ((A*B + C*D)==0)*1e-10)
    ta1 = (alpha+mxn.sqrt(alpha*alpha+4))/2
    ta2 = (alpha-mxn.sqrt(alpha*alpha+4))/2
    ra1 = mxn.arctan(ta1)
    ra2 = mxn.arctan(ta2)+3.14159265358979
    return ra1,ra2

def get_lam_rot(A,B,C,D,alpha,beta):
  return -A*mxn.cos(alpha)*mxn.sin(beta)+B*mxn.sin(alpha)*mxn.sin(beta)-C*mxn.cos(alpha)*mxn.cos(beta)+D*mxn.sin(alpha)*mxn.cos(beta)

def solver_rot(A,B,C,D,alpha,beta0,flag=1):

  tb = (A*mxn.sin(alpha) + B*mxn.cos(alpha))/(C*mxn.sin(alpha) + D*mxn.cos(alpha) + ((C*mxn.sin(alpha) + D*mxn.cos(alpha))==0)*1e-10 + (mxn.abs(C*mxn.sin(alpha) + D*mxn.cos(alpha))<1e-10)*1e-10*(mxn.sign(C*mxn.sin(alpha) + D*mxn.cos(alpha))))
  beta=mxn.arctan(tb)
  beta = beta + (beta<0)*3.14159265358979
  lam = get_lam_rot(A,B,C,D,alpha,beta)

  return beta, 1.0*(lam<=0)*(beta<=beta0)

def point_rot(n1,n2,a):
  return n1*mxn.cos(a) + n2*mxn.sin(a)

def distance(P1, P2, Y):
    #print(P1[0:2,:,:],P2[0:2,:,:])
    dis = mxn.sqrt(mxn.sum((P1-P2)*(P1-P2), axis = 1)+1e-20)
    #print(dis)
    return mxn.min((dis*Y + 1000*(1-Y)), axis = 0)

def checker_rot(A,B,C,D,E,F,G,H,alpha,beta):

  lam1 = get_lam_rot(A,B,C,D,alpha,beta)
  lam2 = get_lam_rot(E,F,G,H,beta,alpha)

  return 1.0*(lam1<=0)*(lam2<=0)
  
def cases_rot(X1, X2, X3, X4, N1, N2, N3, N4, A, B, C, D, alpha0, beta0, batch, dim):

      '---CASE 1: g1 =0 (alpha =0), lam2 =0, lam3 =0, lam4 =0----'
      beta, y = solver_rot(A, B, C, D, mxn.array([0]), beta0,flag=0)
      P1 = mxn.expand_dims(X1,axis=0)
      P2 = mxn.expand_dims(point_rot(N3, N4, beta),axis=0)
      Y = mxn.expand_dims(y,axis=0)
      
      '---CASE 2: g2 = 0 (alpha =alpha0), lam1 =0, lam3 =0, lam4 =0----'
      beta, y = solver_rot(-A, -B, -C, -D, alpha0, beta0,flag=0)
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(X2,axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(point_rot(N3, N4, beta),axis=0),dim=0)
      
      '---CASE 3: g3 = 0 (beta =0), lam1 =0, lam2 =0, lam4 =0----'
      alpha, y = solver_rot(A, C, B, D, mxn.array([0]), alpha0)
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(point_rot(N1, N2, alpha),axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(X3,axis=0),dim=0)
      
      '---CASE 4: g4 = 0 (beta =beta0), lam1 =0, lam2 =0, lam3 =0----'
      alpha, y = solver_rot(-A, -C, -B, -D, beta0, alpha0)
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(X4,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(point_rot(N1, N2, alpha),axis=0),dim=0)
      
      '---CASE 5: g1 =0 (alpha =0), g3 = 0 (beta =0), lam2 =0, lam4 =0----'
      y = checker_rot(A, B, C, D, A, C, B, D, mxn.array([0]), mxn.array([0]))
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(X1,axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(X3,axis=0),dim=0)
      
      '---CASE 6: g1 =0 (alpha =0), g4 = 0 (beta =beta0), lam2 =0, lam3 =0----'
      y = checker_rot(A, B, C, D, -A, -C, -B, -D, mxn.array([0]), beta0)
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(X1,axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(X4,axis=0),dim=0)
      
      '---CASE 7: g2 =0 (alpha =alpha0), g3 = 0 (beta =0), lam1 =0, lam4 =0----'
      y = checker_rot(-A, -B, -C, -D, A, C, B, D, alpha0, mxn.array([0]))
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(X2,axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(X3,axis=0),dim=0)
      
      '---CASE 8: g2 =0 (alpha =alpha0), g4 = 0 (beta =beta0), lam1 =0, lam3 =0----'
      y = checker_rot(-A, -B, -C, -D, -A, -C, -B, -D, alpha0, beta0)
      Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
      P1 = mxn.concat(P1, mxn.expand_dims(X2,axis=0),dim=0)
      P2 = mxn.concat(P2, mxn.expand_dims(X4,axis=0),dim=0)
      
      return P1, P2, Y

def opt_pts_rot(X1, X2, X3, X4,batch,dim):

    N1, N2 = getn(X1, X2)
    N3, N4 = getn(X3, X4)

    a0 = mxn.arccos(mxn.sum(X1*X2, axis = 0))
    b0 = mxn.arccos(mxn.sum(X3*X4, axis = 0))
    #print(mxn.min(a0))
    #print(mxn.min(b0))

    A = -mxn.sum(N2*N4, axis = 0)
    B = -mxn.sum(N1*N4, axis = 0)
    C = -mxn.sum(N2*N3, axis = 0)
    D = -mxn.sum(N1*N3, axis = 0)
    #print(A,B,C,D)

    P1, P2, Y = cases_rot(X1, X2, X3, X4, N1, N2, N3, N4, A, B, C, D, a0, b0, batch, dim)

    ra1, ra2 = get_roots(A, B, C, D)

    rb1 = mxn.arctan((A*mxn.sin(ra1) + B*mxn.cos(ra1))/(C*mxn.sin(ra1) + D*mxn.cos(ra1) + ((C*mxn.sin(ra1) + D*mxn.cos(ra1))==0)*1e-10 + (mxn.abs(C*mxn.sin(ra1) + D*mxn.cos(ra1))<1e-10)*1e-10*(mxn.sign(C*mxn.sin(ra1) + D*mxn.cos(ra1)))))
    rb1 = rb1 + (rb1<0)*3.14159265358979

    rb2 = mxn.arctan((A*mxn.sin(ra2) + B*mxn.cos(ra2))/(C*mxn.sin(ra2) + D*mxn.cos(ra2) + ((C*mxn.sin(ra2) + D*mxn.cos(ra2))==0)*1e-10 + (mxn.abs(C*mxn.sin(ra2) + D*mxn.cos(ra2))<1e-10)*1e-10*(mxn.sign(C*mxn.sin(ra2) + D*mxn.cos(ra2)))))
    rb2 = rb2 + (rb2<0)*3.14159265358979

    P1 = mxn.concat(P1, mxn.expand_dims(point_rot(N1, N2, ra1),axis=0),dim=0)
    P2 = mxn.concat(P2, mxn.expand_dims(point_rot(N3, N4, rb1),axis=0),dim=0)
    y = 1.0*(ra1<=a0)*(rb1<=b0)
    Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)

    P1 = mxn.concat(P1, mxn.expand_dims(point_rot(N1, N2, ra2),axis=0),dim=0)
    P2 = mxn.concat(P2, mxn.expand_dims(point_rot(N3, N4, rb2),axis=0),dim=0)
    y = 1.0*(ra2<=a0)*(rb2<=b0)
    Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)
    #print(Y)

    dist = distance(P1, P2, Y) #shape (batch,)

    return dist
