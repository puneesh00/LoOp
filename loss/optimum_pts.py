import numpy as np
import mxnet as mx

def normalize(X1, X2, X3, X4):
    X1 = np.divide(X1, np.sqrt(np.sum(np.square(X1), axis = 0))) # X1 = X1/np.sqrt(np.sum(np.square(X1), axis = 0))
    X2 = np.divide(X2, np.sqrt(np.sum(np.square(X2), axis = 0))) # X2 = X2/np.sqrt(np.sum(np.square(X2), axis = 0))
    X3 = np.divide(X3, np.sqrt(np.sum(np.square(X3), axis = 0))) # X3 = X3/np.sqrt(np.sum(np.square(X3), axis = 0))
    X4 = np.divide(X4, np.sqrt(np.sum(np.square(X4), axis = 0))) # X4/np.sqrt(np.sum(np.square(X4), axis = 0))
    return X1, X2, X3, X4

def getn(X1, X2):
    N1 = X1
    dotp  = X1*np.einsum('ij,ij->j', X1, X2)
    N2 = X2 - dotp
    N2 = N2/np.sqrt(np.einsum('ij, ij->j', N2, N2))
    return N1, N2

def get_roots(A, B, C, D):
    alpha = (A*A - B*B + C*C - D*D)/(A*B + C*D)
    ta1 = (alpha+np.sqrt(alpha*alpha+4))/2
    ta2 = (alpha-np.sqrt(alpha*alpha+4))/2
    ra1 = np.arctan(ta1)
    ra2 = np.arctan(ta2)+np.pi
    return ra1,ra2

def get_lam_rot(A,B,C,D,alpha,beta):
  return -A*np.cos(alpha)*np.sin(beta)+B*np.sin(alpha)*np.sin(beta)-C*np.cos(alpha)*np.cos(beta)+D*np.sin(alpha)*np.cos(beta)

def solver_rot(A,B,C,D,alpha,beta0,flag=1):

  tb = (A*np.tan(alpha) + B)/(C*np.tan(alpha) + D)
  beta=np.arctan(tb)
  beta = beta + (beta<0)*np.pi
  lam = get_lam_rot(A,B,C,D,alpha,beta)

  return beta, 1*(lam<=0)*(beta<=beta0)
  '''
  if beta<=beta0 and lam<=0:
    return beta, np.ones(np.size(beta))
  else:
    return 1000, np.zeros(np.size(beta))
  '''

def point_rot(n1,n2,a):
  return n1*np.cos(a) + n2*np.sin(a)

def distance(P1, P2, Y):
    #print(P1[0:2,:,:],P2[0:2,:,:])
    dis =np.sqrt(np.sum((P1-P2)*(P1-P2), axis = 1))
    #print(dis)
    return np.min((dis*Y + 1000*(1-Y)), axis = 0)

def checker_rot(A,B,C,D,E,F,G,H,alpha,beta):

  lam1 = get_lam_rot(A,B,C,D,alpha,beta)
  lam2 = get_lam_rot(E,F,G,H,beta,alpha)

  return 1*(lam1<=0)*(lam2<=0)
  '''
  if lam1<=0 and lam2<=0:
    return True
  else:
    return False
  '''
def cases_rot(X1, X2, X3, X4, N1, N2, N3, N4, A, B, C, D, alpha0, beta0, batch, dim):

      P1 = np.zeros((10, dim,batch))
      P2 = np.zeros((10, dim,batch))
      Y = np.zeros((10, batch))
      '---CASE 1: g1 =0 (alpha =0), lam2 =0, lam3 =0, lam4 =0----'
      beta, y = solver_rot(A, B, C, D, 0, beta0,flag=0)
      P1[0,:,:] = X1
      P2[0,:,:] = point_rot(N3, N4, beta)
      Y[0,:] = y
      #dist[0,:] = np.sum((P1-P2)*(P1-P2), axis = 0)*y + 1000*(1-y)
      '''
      if y:
       p1=x1
       p2=point_rot(n3,n4,beta)
       print('yes1')
       return p1,p2
      '''
      '---CASE 2: g2 = 0 (alpha =alpha0), lam1 =0, lam3 =0, lam4 =0----'
      beta, y = solver_rot(-A, -B, -C, -D, alpha0, beta0,flag=0)
      Y[1,:] = y
      P1[1,:,:] = X2
      P2[1,:,:] = point_rot(N3, N4, beta)
      #dist[1,:] = np.sum((P1-P2))
      '''
      if y:
        p1=x2
        p2=point_rot(n3,n4,beta)
        print('yes2')
        return p1,p2
      '''
      '---CASE 3: g3 = 0 (beta =0), lam1 =0, lam2 =0, lam4 =0----'
      alpha, y = solver_rot(A, C, B, D, 0, alpha0)
      Y[2,:] = y
      P1[2,:,:] = point_rot(N1, N2, alpha)
      P2[2,:,:] = X3
      '''
      if y:
        p2=x3
        p1=point_rot(n1,n2,alpha)
        print('yes3')
        return p1,p2
      '''
      '---CASE 4: g4 = 0 (beta =beta0), lam1 =0, lam2 =0, lam3 =0----'
      alpha, y = solver_rot(-A, -C, -B, -D, beta0, alpha0)
      Y[3,:] = y
      P2[3,:,:] = X4
      P1[3,:,:] = point_rot(N1, N2, alpha)
      '''
      if y:
        p2=x4
        p1=point_rot(n1,n2,alpha)
        print('yes4')
        return p1,p2
      '''
      '---CASE 5: g1 =0 (alpha =0), g3 = 0 (beta =0), lam2 =0, lam4 =0----'
      y = checker_rot(A, B, C, D, A, C, B, D, 0, 0)
      Y[4,:] = y
      P1[4,:,:] = X1
      P2[4,:,:] = X3
      '''
      if y:
        p1=x1
        p2=x3
        print('yes5')
        return p1,p2
      '''
      '---CASE 6: g1 =0 (alpha =0), g4 = 0 (beta =beta0), lam2 =0, lam3 =0----'
      y = checker_rot(A, B, C, D, -A, -C, -B, -D, 0, beta0)
      Y[5,:] = y
      P1[5,:,:] = X1
      P2[5,:,:] = X4
      '''
      if y:
        p1=x1
        p2=x4
        print('yes6')
        return p1,p2
      '''
      '---CASE 7: g2 =0 (alpha =alpha0), g3 = 0 (beta =0), lam1 =0, lam4 =0----'
      y = checker_rot(-A, -B, -C, -D, A, C, B, D, alpha0, 0)
      Y[6,:] = y
      P1[6,:,:] = X2
      P2[6,:,:] = X3
      '''
      if y:
        p1=x2
        p2=x3
        print('yes7')
        return p1,p2
      '''
      '---CASE 8: g2 =0 (alpha =alpha0), g4 = 0 (beta =beta0), lam1 =0, lam3 =0----'
      y = checker_rot(-A, -B, -C, -D, -A, -C, -B, -D, alpha0, beta0)
      Y[7,:] = y
      P1[7,:,:] = X2
      P2[7,:,:] = X4
      '''
      if y:
        p1=x2
        p2=x4
        print('yes8')
        return p1,p2
      #print('no')
      return p1,p2
      '''
      return P1, P2, Y

def opt_pts_rot(X1, X2, X3, X4,batch,dim):

    N1, N2 = getn(X1, X2)
    N3, N4 = getn(X3, X4)

    a0 = np.arccos(np.sum(X1*X2, axis = 0))
    b0 = np.arccos(np.sum(X3*X4, axis = 0))
    print(a0,b0)

    A = -np.sum(N2*N4, axis = 0)
    B = -np.sum(N1*N4, axis = 0)
    C = -np.sum(N2*N3, axis = 0)
    D = -np.sum(N1*N3, axis = 0)
    print(A,B,C,D)

    P1, P2, Y = cases_rot(X1, X2, X3, X4, N1, N2, N3, N4, A, B, C, D, a0, b0, batch, dim)

    ra1, ra2 = get_roots(A, B, C, D)

    rb1 = np.arctan((A*np.tan(ra1) + B)/(C*np.tan(ra1) + D))
    rb1 = rb1 + (rb1<0)*np.pi

    rb2 = np.arctan((A*np.tan(ra2) + B)/(C*np.tan(ra2) + D))
    rb2 = rb2 + (rb2<0)*np.pi

    P1[8,:,:] = point_rot(N1, N2, ra1)
    P2[8,:,:] = point_rot(N3, N4, rb1)
    Y[8,:] = 1*(ra1<=0)*(rb1<=b0)

    P1[9,:,:] = point_rot(N1, N2, ra2)
    P2[9,:,:] = point_rot(N3, N4, rb2)
    Y[9,:] = 1*(ra2<=0)*(rb2<=b0)
    print(Y)

    dist = distance(P1, P2, Y) #shape (batch,)

    return dist
'''
X1=[[0.33608175, 0.83448615, 0.43667142], [0.89592573, 0.31949744, 0.30860731]]
X2=[[0.40752846, 0.74777028, 0.5241757 ], [0.85835948, 0.49176357, 0.14624495]]
X3=[[0.50765006, 0.84737936, 0.15569086], [0.6206815, 0.58409488, 0.52305606]]
X4=[[0.53881,    0.60703189, 0.58411992], [0.83163155, 0.1047268,  0.54536342]]

X1=np.transpose(np.asarray(X1))
X2=np.transpose(np.asarray(X2))
X3=np.transpose(np.asarray(X3))
X4=np.transpose(np.asarray(X4))

X1=mx.nd.array(X1)
X2=mx.nd.array(X2)
X3=mx.nd.array(X3)
X4=mx.nd.array(X4)
print(X1)
s=X1.shape



dis=opt_pts_rot(np.squeeze(np.asarray(X1.as_np_ndarray(),dtype='float32')),np.squeeze(np.asarray(X2.as_np_ndarray(),dtype='float32')),np.squeeze(np.asarray(X3.as_np_ndarray(),dtype='float32')),np.squeeze(np.asarray(X4.as_np_ndarray(),dtype='float32')),s[1],s[0])
print(dis)
dis=mx.nd.array(np.squeeze(np.asarray(dis)))
print(dis)
'''