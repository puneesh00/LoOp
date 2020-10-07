import mxnet.ndarray as mxn

def similarity(P1, P2, Y):
    dis = mxn.sum(P1*P2, axis = 1)
    return -mxn.min((-dis*Y + 1000*(1-Y)), axis = 0)


def point_lin(x1,x2,cos):
	po1 = x1*(1-cos)*0.5 + x2*(1+cos)*0.5
	return po1


def get_lam_lin(A,B,C,kp,k):
  return -C - B*kp -A*k


def solver_lin(A, Ap, B, Bp, C, Cp,k):

  kp = -(Ap*k+Cp)/(Bp+(mxn.abs(Bp)<1e-10)*1e-10*mxn.sign(Bp)+(Bp==0)*1e-10)
  lam = get_lam_lin(A,B,C,kp,k)

  return kp, 1.0*(kp>=0)*(lam<=0)


def checker_lin(A, Ap, B, Bp, C, Cp, k, kp):

  lam1 = get_lam_lin(A,B,C,kp,k)
  lam2 = get_lam_lin(Ap,Bp,Cp,kp,k)
  
  return 1.0*(lam1<=0)*(lam2<=0)


def cases_lin(x1,x2,x3,x4):

  A = mxn.sum((x1-x2)*(x1-x2), axis=0)
  B = mxn.sum((x3-x4)*(x2-x1), axis=0)
  C = mxn.sum((x1-x3)*(x2-x1), axis=0)

  Ap = mxn.sum((x2-x1)*(x3-x4), axis=0)
  Bp = mxn.sum((x3-x4)*(x3-x4), axis=0)
  Cp = mxn.sum((x1-x3)*(x3-x4), axis=0)

  '---CASE 1: g1 =0 (k =0), lam2 =0, lam3 =0, lam4 =0----'
  kp, y = solver_lin(A, Ap, B, Bp, C, Cp, 0)
  p1 = mxn.expand_dims(x1,axis=0)
  p2 = mxn.expand_dims(point_lin(x3, x4, (2*kp-1)),axis=0)
  Y = mxn.expand_dims(y,axis=0)

  '---CASE 2: g2 = 0 (k =1), lam1 =0, lam3 =0, lam4 =0----'
  kp, y = solver_lin(-A, Ap, -B, Bp, -C, Cp, 1)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(x2,axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(point_lin(x3, x4, (2*kp-1)),axis=0),dim=0)

  '---CASE 3: g3 = 0 (kp =0), lam1 =0, lam2 =0, lam4 =0----'
  k, y = solver_lin(Bp, B, Ap, A, Cp, C, 0)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(point_lin(x1, x2, (2*k-1)),axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(x3,axis=0),dim=0)

  '---CASE 4: g4 = 0 (k =1), lam1 =0, lam2 =0, lam3 =0----'
  k, y = solver_lin(-Bp, B, -Ap, A, -Cp, C, 1)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(point_lin(x1, x2, (2*k-1)),axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(x4,axis=0),dim=0)

  '---CASE 5: g1 =0 (k =0), g3 = 0 (kp =0), lam2 =0, lam4 =0----'
  y = checker_lin(A, Ap, B, Bp, C, Cp, 0,0)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(x1,axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(x3,axis=0),dim=0) 

  '---CASE 6: g1 =0 (k =0), g4 = 0 (kp =1), lam2 =0, lam3 =0----'
  y = checker_lin(A, -Ap, B, -Bp, C, -Cp, 0,1)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(x1,axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(x4,axis=0),dim=0) 

  '---CASE 7: g2 =0 (k =1), g3 = 0 (kp =0), lam1 =0, lam4 =0----'
  y = checker_lin(-A, Ap, -B, Bp, -C, Cp, 1,0)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(x2,axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(x3,axis=0),dim=0)  

  '---CASE 8: g2 =0 (k =1), g4 = 0 (kp =1), lam1 =0, lam3 =0----'
  y = checker_lin(-A, -Ap, -B, -Bp, -C, -Cp, 1,1)
  Y = mxn.concat(Y, mxn.expand_dims(y,axis=0),dim=0)
  p1 = mxn.concat(p1, mxn.expand_dims(x2,axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(x4,axis=0),dim=0)  

  return p1,p2,Y


def opt_pts_line(x1,x2,x3,x4):
  t=time.time()

  a = x1-x2
  b = x1+x2
  c = x3-x4
  d = x3+x4
  q = b-d

  asq = mxn.sum(a*a, axis=0)
  csq = mxn.sum(c*c, axis=0)
  ac = mxn.sum(a*c, axis=0)
  aq = mxn.sum(a*q, axis=0)
  cq = mxn.sum(c*q, axis=0)

  Dr = ac**2 - asq*csq

  c_alpha = -csq*aq + ac*cq
  c_alpha = c_alpha/(Dr+(mxn.abs(Dr)<1e-10)*1e-10*mxn.sign(Dr)+(Dr==0)*1e-10)

  c_beta = asq*cq - ac*aq
  c_beta = c_beta/(Dr+(mxn.abs(Dr)<1e-10)*1e-10*mxn.sign(Dr)+(Dr==0)*1e-10)

  p1,p2,Y = cases_lin(x1,x2,x3,x4)

  p1 = mxn.concat(p1, mxn.expand_dims(point_lin(x1, x2, c_alpha),axis=0),dim=0)
  p2 = mxn.concat(p2, mxn.expand_dims(point_lin(x3, x4, c_beta),axis=0),dim=0)
  y = 1.0*(c_alpha>=-1)*(c_alpha<=1)*(c_beta>=-1)*(c_beta<=1)
  Y = mxn.concat(Y,mxn.expand_dims(y,axis=0),dim=0)

  dist = similarity(p1, p2, Y)

  return dist
