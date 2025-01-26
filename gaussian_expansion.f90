module gaussian_expansion
implicit none
save

real(8), parameter :: pi = 3.14159265359d0

contains

!----------------------------------------------------------------------------------------!

!Numerical Recipes
!Gauss Legendre grid points and weights
subroutine gaulegf(x1, x2, x, w, n)
  integer, intent(in) :: n
  real(8), intent(in) :: x1, x2
  real(8), dimension(n), intent(out) :: x, w
  integer :: i, j, m
  real(8) :: p1, p2, p3, pp, xl, xm, z, z1
  real(8), parameter :: eps=3.d-14
      
  pp = 0.0d0
  m = (n+1)/2
  xm = 0.5d0*(x2+x1)
  xl = 0.5d0*(x2-x1)
  do i=1,m
    z = cos(3.141592654d0*(i-0.25d0)/(n+0.5d0))
    z1 = 0.0
    do while(abs(z-z1) .gt. eps)
      p1 = 1.0d0
      p2 = 0.0d0
      do j=1,n
        p3 = p2
        p2 = p1
        p1 = ((2.0d0*j-1.0d0)*z*p2-(j-1.0d0)*p3)/j
      end do
      pp = n*(z*p1-p2)/(z*z-1.0d0)
      z1 = z
      z = z1 - p1/pp
    end do
    x(i) = xm - xl*z
    x(n+1-i) = xm + xl*z
    w(i) = (2.0d0*xl)/((1.0d0-z*z)*pp*pp)
    w(n+1-i) = w(i)
  end do

end subroutine gaulegf

!----------------------------------------------------------------------------------------!

!Numerical Recipes
!Associated Legendre polynomial
real(8) function plgndr(l, m, x)
  integer, intent(in) :: l, m
  real(8), intent(in) :: x
  integer :: i, ll
  real(8) :: fact, oldfact, pll, pmm, pmmp1, omx2
  real(8), parameter :: pi = 3.14159265359d0
  
  pll = 0.0d0
  if(m.lt.0.or.m.gt.l.or.abs(x).gt.1.) print *, 'bad arguments in plgndr'
  
  pmm=1.0d0
  if (m .gt. 0) then
    omx2 = (1.d0-x)*(1.d0+x)
    fact = 1.d0
    do i = 1, m
      pmm = pmm*omx2*fact/(fact+1.d0)
      fact = fact + 2.d0
    end do
  end if
  
  pmm = sqrt((2*m + 1)*pmm/(4.d0*pi))
  if (mod(m, 2) .eq. 1) pmm = -pmm
  if (l .eq. m) then
    plgndr = pmm
  else
    pmmp1 = x*sqrt(2.d0*m + 3.d0)*pmm
    if (l .eq. m+1) then
      plgndr = pmmp1
    else
      oldfact=sqrt(2.d0*m + 3.d0)
    do ll = m+2, l
      fact = sqrt((4.d0*ll**2 - 1.d0)/(ll**2-m**2))
      pll = (x*pmmp1-pmm/oldfact)*fact
      oldfact = fact
      pmm = pmmp1
      pmmp1 = pll
    end do
      plgndr = pll
    end if
  end if

end function plgndr

!----------------------------------------------------------------------------------------!

!Hyperbolic cotangent in quad precision
real(kind=16) function cothquad(x)
  real(kind=16), intent(in) :: x
  
  cothquad = (1.0e0_16 + exp(-2.0e0_16*x))/(1.0e0_16 - exp(-2.0e0_16*x))

end function cothquad

!----------------------------------------------------------------------------------------!

!Partial expansion in quad precision (required for numerical stability)
!This depends only on the distance of the Gaussian from the origin
subroutine fsubquad(r2, sigma2, rij2, lmax, f2)
  real(kind=8), intent(in) :: r2, sigma2, rij2
  real(kind=8), dimension(0:lmax), intent(out) :: f2
  integer, intent(in) :: lmax
  real(kind=16) :: r, sigma, rij
  real(kind=16), dimension(0:lmax) :: f
  real(kind=16) :: alpha, frev
  real(kind=16), parameter :: eps = 1.0e-5_16
  integer :: i

  r = real(r2, 16)
  sigma = real(sigma2, 16)
  rij = real(rij2, 16)
  alpha = rij/sigma**2
  f = 0.0e0_16

  if (abs(r - rij) >= 5.0_16*sigma) then
    f2 = dble(f)
    return
  end if

  if (r <= eps) then
    f2 = dble(f)
    return
  end if

  if (rij <= eps) then
    f(0) = exp(-r**2/(2.0e0_16*sigma**2))*r
    f2 = dble(f)
    return
  end if

  !l=0
  f(0) = exp(-0.5e0_16*(r - rij)**2/sigma**2)
  f(0) = f(0) - exp(-0.5e0_16*(r + rij)**2/sigma**2)
  f(0) = 0.5e0_16*f(0)/alpha

  if (lmax == 0) then
    f2 = dble(f)
    return
  end if

  !l=1
  if (alpha*r >= eps) then !exact
    f(1) = f(0)*(cothquad(alpha*r) - 1.0e0_16/(r*alpha))
  else !first three terms in Taylor series
    f(1) = f(0)*(alpha*r/3.0e0_16 - (alpha*r)**3/45.0e0_16 + 2.0e0_16*(alpha*r)**5/945.0e0_16)
  end if

  !recurrence relation for modified spherical Bessel functions
  !in = in-1 - (2(n-1) + 1)in-1/x
  do i = 2, lmax
    f(i) = f(i-2) - (2.0e0_16*real(i - 1, 16) + 1.0e0_16)*f(i-1)/(alpha*r)
    frev = f(i) + (2.0e0_16*real(i - 1, 16) + 1.0e0_16)*f(i-1)/(alpha*r)
    frev = f(i-1) + (2.0e0_16*real(i - 2, 16) + 1.0e0_16)*frev/(alpha*r)
    if (i >= 3 .and. abs(frev - f(i-3)) >= 1.0e-100_16) then
      if (f(i)/f(0) > 1.0e-8_16) cycle
      f(i:lmax) = 0.0d0
      exit
    end if
  end do

  f2 = dble(f)

end subroutine fsubquad

!----------------------------------------------------------------------------------------!

!Full expansion
!This depends on the displacement (distance and angles) of the Gaussian from the origin
subroutine f2sub(f, nmax, lmax, cost, f2)
  real(8), dimension(0:nmax, 0:lmax), intent(in) :: f
  real(8), intent(in) :: cost
  integer, intent(in) :: nmax, lmax
  real(8), dimension(0:nmax, 0:lmax, 0:lmax), intent(out) :: f2
  integer :: i, j

  do i = 0, lmax
    do j = 0, i
      f2(:, i, j) = f(:, i)*plgndr(i, j, cost)
    end do
  end do

end subroutine f2sub

!----------------------------------------------------------------------------------------!

end module gaussian_expansion
