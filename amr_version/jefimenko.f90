! Compute electric field according to Jefimenko's equations
!
! Author: Jannis Teunissen
program jefimenko
  use m_config

  implicit none

  integer, parameter  :: dp             = kind(0.0d0)
  real(dp), parameter :: pi             = 3.141592653589793_dp
  real(dp), parameter :: speed_of_light = 299792458.0_dp ! m/s
  real(dp), parameter :: epsilon_0      = 8.8541878128e-12_dp ! SI units

  type :: coords_t
     real(dp), allocatable :: x(:)
  end type coords_t

  type grid_t
     integer               :: n_dims
     integer               :: dims(3) = 1
     integer               :: min_index(3) = 1
     integer               :: max_index(3) = 1
     type(coords_t)        :: coords_fc(3)
     type(coords_t)        :: coords_cc(3)
     real(dp), allocatable :: values(:, :, :)
     real(dp)              :: dr(3) = 1.0_dp
  end type grid_t

  type domain_t
     integer  :: cycle
     real(dp) :: time
  end type domain_t

  type input_t
     type(grid_t), allocatable :: d1_rhs(:) ! d/dt rhs
     type(grid_t), allocatable :: d2_rhs(:) ! (d/dt)^2 rhs
     type(grid_t), allocatable :: d1_Jx(:)  ! d/dt Jx
     type(grid_t), allocatable :: d2_Jx(:)  ! (d/dt)^2 Jx 
     type(grid_t), allocatable :: d1_Jy(:)
     type(grid_t), allocatable :: d2_Jy(:)
     type(grid_t), allocatable :: d1_Jz(:)
     type(grid_t), allocatable :: d2_Jz(:)
     type(domain_t) :: domain
  end type input_t

  type(CFG_t)        :: cfg
  type(input_t)      :: input
  integer            :: i, i_start = -1, i_end = -1
  integer            :: my_unit
  character(len=200) :: raw_basename = "UNDEFINED"
  character(len=200) :: output_basename = "output"
  character(len=200) :: fname, outname
  real(dp)           :: r_obs(3) = [1e3_dp, 0.0_dp, 0.0_dp]
  real(dp)           :: E_static(3), E_rho(3), E_J(3)
  real(dp)           :: r_min(3), r_max(3)
  real(dp)           :: r_center(3), distance_from_center, delay_from_center

  call CFG_update_from_arguments(cfg)

  call CFG_add_get(cfg, "raw_basename", raw_basename, &
       "Base of filename of raw files")
  call CFG_add_get(cfg, "output_basename", output_basename, &
       "Base name of output text file")
  call CFG_add_get(cfg, "i_start", i_start, "Start index")
  call CFG_add_get(cfg, "i_end", i_end, "End index")

  if (i_start < 0) error stop "i_start < 0"
  if (i_end < i_start) error stop "i_end < i_start"

  call CFG_add_get(cfg, "r_obs", r_obs, "Observation location (m)")
  call CFG_check(cfg)

  write(outname, "(A,A,E0.3,A,E0.3,A,E0.3,A)") trim(output_basename), &
       "_observer_", r_obs(1), "_", r_obs(2), "_", r_obs(3), ".txt"

  delay_from_center = 0.0_dp

  open(newunit=my_unit, file=trim(outname), action="write")
  write(my_unit, "(A)") 't_obs,t_src,E_static_x,E_static_y,E_static_z,E_rho_x,'&
       &'E_rho_y,E_rho_z,E_J_x,E_J_y,E_J_z'

  do i = i_start, i_end
     write(*, "(A,I0,A,I0)") "At index: ", i, "/", i_end
     ! Read d/dt rho and d/dt (Jx, Jy, Jz)
     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d1_rhs_", i, ".raw"
     call read_raw_file(trim(fname), input%d1_rhs, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d2_rhs_", i, ".raw"
     call read_raw_file(trim(fname), input%d2_rhs, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d1_Je_1_", i, ".raw"
     call read_raw_file(trim(fname), input%d1_Jx, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d2_Je_1_", i, ".raw"
     call read_raw_file(trim(fname), input%d2_Jx, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d1_Je_2_", i, ".raw"
     call read_raw_file(trim(fname), input%d1_Jy, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d2_Je_2_", i, ".raw"
     call read_raw_file(trim(fname), input%d2_Jy, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d1_Je_3_", i, ".raw"
     call read_raw_file(trim(fname), input%d1_Jz, input%domain)

     write(fname, "(A,A,I0.6,A)") trim(raw_basename), "_d2_Je_3_", i, ".raw"
     call read_raw_file(trim(fname), input%d2_Jz, input%domain)

     if (i == i_start) then
        ! Determine center of domain
        call get_rmin_rmax(input%d1_rhs, r_min, r_max)
        r_center = 0.5_dp * (r_min + r_max)

        distance_from_center = norm2(r_obs - r_center)
        delay_from_center = distance_from_center/speed_of_light
     end if

     call compute_Efield(input, E_rho, E_J)
     E_static(:) = 0.0_dp

     write(my_unit, fmt='(E14.6,10(",",E14.6))') &
          input%domain%time + delay_from_center, &
          input%domain%time, E_static, E_rho, E_J
  end do

  close(my_unit)
  write(*, "(A,A)") "Output written to ", trim(outname)

contains

  subroutine compute_Efield(input, E_rho, E_J)
    type(input_t), intent(in) :: input
    real(dp)                  :: E_rho(3), E_J(3)
    integer                   :: n, i, j, k
    integer                   :: ilo(3), ihi(3)
    real(dp)                  :: dV, R, one_over_R, t_corr, factor
    real(dp)                  :: r_cc(3), r_diff(3), r_hat(3)
    real(dp)                  :: E_rho_tmp(3), E_J_tmp(3)

    real(dp), parameter :: f0 = 1 / (4 * pi * epsilon_0 * speed_of_light)

    E_rho(:) = 0.0_dp
    E_J(:)   = 0.0_dp

    do n = 1, size(input%d1_rhs)
       ! Volume of grid cell
       dV = product(input%d1_rhs(n)%dr)

       ! To reduce round-off error
       E_rho_tmp(:) = 0.0_dp
       E_J_tmp(:)   = 0.0_dp

       ilo = input%d1_rhs(n)%min_index
       ihi = input%d1_rhs(n)%max_index
       do k = ilo(3), ihi(3)
          r_cc(3) = input%d1_rhs(n)%coords_cc(3)%x(k)
          do j = ilo(2), ihi(2)
             r_cc(2) = input%d1_rhs(n)%coords_cc(2)%x(j)
             do i = ilo(1), ihi(1)
                r_cc(1) = input%d1_rhs(n)%coords_cc(1)%x(i)

                r_diff = r_obs - r_cc
                R = norm2(r_diff)
                one_over_R = 1/R
                r_hat = r_diff * one_over_R
                factor = f0 * dV * one_over_R

                ! Correction for retarded time w.r.t. center of domain
                t_corr = (distance_from_center - R)/speed_of_light

                E_rho_tmp = E_rho_tmp + factor * (-epsilon_0) * r_hat * &
                     (input%d1_rhs(n)%values(i, j, k) + &
                     t_corr * input%d2_rhs(n)%values(i, j, k))
                E_J_tmp = E_J_tmp - factor/speed_of_light * &
                     [input%d1_Jx(n)%values(i, j, k) + t_corr * &
                     input%d2_Jx(n)%values(i, j, k), &
                     input%d1_Jy(n)%values(i, j, k) + t_corr * &
                     input%d2_Jy(n)%values(i, j, k), &
                     input%d1_Jz(n)%values(i, j, k) + t_corr * &
                     input%d2_Jz(n)%values(i, j, k)]
             end do
          end do
       end do

       E_rho = E_rho + E_rho_tmp
       E_J   = E_J + E_J_tmp
    end do
  end subroutine compute_Efield

  subroutine read_raw_file(filename, grids, domain)
    character(len=*), intent(in)             :: filename
    type(grid_t), intent(inout), allocatable :: grids(:)
    type(domain_t), intent(inout)            :: domain

    integer :: my_unit, n, dim, k, n_dims, n_grids, n_cells(3)

    open(newunit=my_unit, file=trim(filename), access="stream", &
         form="unformatted", status="old")
    read(my_unit) domain%cycle
    read(my_unit) domain%time
    read(my_unit) n_grids

    if (allocated(grids)) deallocate(grids)
    allocate(grids(n_grids))

    do n = 1, n_grids
       read(my_unit) grids(n)%n_dims
       n_dims = grids(n)%n_dims
       read(my_unit) grids(n)%dims(1:n_dims)
       read(my_unit) grids(n)%min_index(1:n_dims)
       read(my_unit) grids(n)%max_index(1:n_dims)

       ! Add one for Fortran style indexing
       grids(n)%min_index(1:n_dims) = grids(n)%min_index(1:n_dims) + 1

       do dim = 1, n_dims
          k = grids(n)%dims(dim)
          allocate(grids(n)%coords_fc(dim)%x(k))

          ! Read face-centered coordinates
          read(my_unit) grids(n)%coords_fc(dim)%x

          ! Compute cell-centered coordinates
          allocate(grids(n)%coords_cc(dim)%x(k-1))
          grids(n)%coords_cc(dim)%x = 0.5_dp * (grids(n)%coords_fc(dim)%x(2:) + &
               grids(n)%coords_fc(dim)%x(1:k-1))

          ! Set grid spacing (is supposed to be constant)
          grids(n)%dr(dim) = grids(n)%coords_fc(dim)%x(2) - &
               grids(n)%coords_fc(dim)%x(1)
       end do

       ! Number of cell centers is one less than number of faces
       n_cells(1:n_dims) = grids(n)%dims(1:n_dims) - 1
       n_cells(n_dims+1:) = 1
       allocate(grids(n)%values(n_cells(1), n_cells(2), n_cells(3)))
       read(my_unit) grids(n)%values
    end do

    close(my_unit)
  end subroutine read_raw_file

  subroutine get_rmin_rmax(grids, r_min, r_max)
    type(grid_t), intent(in) :: grids(:)
    real(dp), intent(out)    :: r_min(3), r_max(3)
    integer                  :: n, ilo(3), ihi(3)

    r_min = huge(1.0_dp)
    r_max = -huge(1.0_dp)

    do n = 1, size(grids)
       ilo = grids(n)%min_index
       ihi = grids(n)%max_index
       r_min = min(r_min, [grids(n)%coords_fc(1)%x(ilo(1)), &
            grids(n)%coords_fc(2)%x(ilo(2)), &
            grids(n)%coords_fc(3)%x(ilo(3))])
       r_max = max(r_max, [grids(n)%coords_fc(1)%x(ihi(1)), &
            grids(n)%coords_fc(2)%x(ihi(2)), &
            grids(n)%coords_fc(3)%x(ihi(3))])
    end do
  end subroutine get_rmin_rmax

  ! subroutine volume_integral(grids, vol_int)
  !   type(grid_t), intent(in) :: grids(:)
  !   real(dp), intent(out)    :: vol_int
  !   integer                  :: n, ilo(3), ihi(3)

  !   vol_int = 0.0_dp
  !   do n = 1, size(grids)
  !      ilo = grids(n)%min_index
  !      ihi = grids(n)%max_index
  !      vol_int = vol_int + product(grids(n)%dr) * sum(&
  !           grids(n)%values(ilo(1):ihi(1), ilo(2):ihi(2), ilo(3):ihi(3)))
  !   end do
  ! end subroutine volume_integral

end program jefimenko
