import numpy as np
import VBBinaryLensing as VBB

pi = np.pi

def reldev(q,t,u0,te,t0,fs,fb,ferr,rs,xp,yp):
    pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus=pspl_n_imagesf(t,u0,te,t0)
    bsx=(t-t0)/te
    bsy=u0
    invq1=1.0/(q+1.0)
    #close=0.
    #interm=0.
    #wide=0.
    total=0.
    newout = 0
    #Erdl and Schneider 1993, Dominik 1999
    #limc,limw=topologylimit.limits(q)
    #Separation: Planet from PSPL center
    d=(xp**2+yp**2)**0.5
    #avoid innermost planets
    if d>0.005:
        #Determine offset from magnification origin (VBB)	
        com=np.array([xp/d,yp/d])
        com=invq1*com
        com=d*q*com
        nsvec=np.array([bsx,bsy])-com
        #Determine rotation between shifted source position to put binary y1 axis (symmetry axis) on y2=0
        phi1=np.angle(complex(xp,yp))
        phi2=np.angle(complex(nsvec[0],nsvec[1]))
        phi3=np.angle(complex(bsx,bsy))
        angle=phi2-phi1-phi3+pi*0.5
        angle=-phi1+pi*0.5
        rmatrix=np.matrix([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        bsinmap=np.dot(rmatrix,np.array([bsx,bsy])).T
        #Avoid central region with numerical artefacts, corresponds to d and 1/d degeneracy for a very distant or close target
        newout=0.
        #fspl = fsplmodel(t,u0,te,t0,rs)

#        result = VBB.BinaryMag(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]),rs,0.005)
        result = VBB.BinaryMag0(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]))

        #result = binary_function(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]),rs,0.01))
        delta_chisqr = (fs*result-fs*pspl)**2/ferr**2
        return delta_chisqr,q,xp,yp,d,rs,float(bsinmap[1][0]),float(bsinmap[0][0]),newout
        #print(fs*VBB.BinaryMag(d,q,float(bsinmap[1][0]),float(bsinmap[0][0]),rs,0.005)-fs*pspl)**2,ferr**2)
    else:
        delta_chisqr=0
        newout=-1
        return 0,q,xp,yp,d,rs,1,1,newout


def pspl_n_imagesf(t,u0,te,t0):
    usqr=u0**2+((t-t0)/te)**2
    usqrt4=(usqr+4.)**0.5
    u=(usqr)**0.5
    uplus  = 0.5*(u+usqrt4)
    uminus = 0.5*(u-usqrt4)
    muplus=uplus**2/(uplus*2-uminus**2)
    muminus=uminus**2/(uplus*2-uminus**2)
    spplus=2.*muplus-1.
    spminus=2.*muminus
    prox=(t-t0)/(te*u)
    proy=u0/u
    pspl=(usqr+2.0)/(u*usqrt4)
    return pspl,uplus,uminus,prox,proy,muplus,muminus,spplus,spminus