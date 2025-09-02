 #include <deal.II/base/utilities.h>
  #include <deal.II/base/quadrature_lib.h>
  #include <deal.II/base/function.h>
  #include <deal.II/base/logstream.h>
  #include <deal.II/base/index_set.h>
  #include <deal.II/base/parsed_function.h>
  #include <deal.II/base/parameter_acceptor.h>
  #include <deal.II/base/conditional_ostream.h>
  #include <deal.II/base/timer.h>
  #include <deal.II/lac/vector.h>
  #include <deal.II/lac/full_matrix.h>
  #include <deal.II/lac/dynamic_sparsity_pattern.h>
 #include <deal.II/lac/solver_gmres.h>
  #include <deal.II/lac/affine_constraints.h>
  #include <deal.II/lac/sparsity_tools.h>
  #include <deal.II/lac/sparse_matrix.h>
  #include <deal.II/base/discrete_time.h>



  #include <deal.II/lac/sparse_direct.h>
  #include <deal.II/lac/precondition.h>
  #include <deal.II/lac/linear_operator.h>
#include <deal.II/grid/tria.h>


#include <deal.II/grid/grid_refinement.h>


#include <deal.II/lac/block_sparsity_pattern.h>

  
  #include <deal.II/grid/grid_generator.h>
  #include <deal.II/grid/grid_out.h>
  #include <deal.II/dofs/dof_handler.h>
  #include <deal.II/dofs/dof_tools.h>
   #include <deal.II/dofs/dof_renumbering.h>
  #include <deal.II/fe/fe_q.h>
  #include <deal.II/fe/fe_values.h>
  #include <deal.II/fe/fe_system.h>

  #include <deal.II/numerics/data_out.h>
  #include <deal.II/numerics/vector_tools.h>

  #include <deal.II/numerics/error_estimator.h>
  #include <deal.II/numerics/solution_transfer.h>
  #include <deal.II/numerics/matrix_tools.h>

  #include <deal.II/lac/solver_cg.h>
  
  #include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/solver_control.h>
  #include <deal.II/grid/grid_in.h>
 #include <deal.II/sundials/arkode.h>


  #include <fstream>
  #include <iostream>

namespace IMEX_NS
{
  using namespace dealii;


  template <int dim>
  class InletVelocity: public dealii::Function<dim>
  { public:
      InletVelocity(): dealii::Function<dim>(dim+1){}

      virtual void vector_value(const dealii::Point<dim> &p,
                          dealii::Vector<double> &values) const override
{
  const double y = p[1];
  const double ux = 6.0 * y * (4.1 - y) / (4.1 * 4.1);
  values = 0.0;
  values[0] = ux;
  if constexpr (dim >= 2) values[1] = 0.0;
}
    };

  template <int dim>
  class NavierStokes: public ParameterAcceptor
  {
    public:
      NavierStokes ();
      void run();
    private: 
     

     Triangulation<dim> triangulation;
     FESystem<dim> fe;
     DoFHandler<dim> dof_handler;

     void setup_system(const double time);

     void output_results(const double time,
                         const unsigned int timestep_number,
                         const BlockVector<double> &solution);

      void prepare_for_coarsening_and_refinement(const BlockVector<double> &solution);
      void transfer_solution_vectors_to_new_mesh(const double time, const std::vector<BlockVector<double>> &all_in, std::vector<BlockVector<double>> &all_out); 
      void setup_arkode( SUNDIALS::ARKode<BlockVector<double>> &arkode);
      void assemble_matrix(const double time);
      void assemble_convective_rhs_from(const BlockVector<double> &y,
                                       BlockVector<double> &rhs);
      
      void update_current_constraints(const double time);

      AffineConstraints<double> hanging_node_constraints;
      AffineConstraints<double> current_constraints;
      AffineConstraints<double> homogeneous_constraints;
      AffineConstraints<double> pressure_boundary_constraints;

      BlockSparsityPattern sparsisty_pattern_;



      

      SUNDIALS::ARKode<dealii::BlockVector<double>>::AdditionalData arkode_data;


      double final_time;
      double initial_time;
      double time_interval_output;
      double abs_tol;
      double rel_tol;
      bool is_linear;
      bool is_time_ind;
      unsigned int max_nonlinear_iterations;
      unsigned int max_order_arkode;
      double minimum_step_size;
      

      unsigned int initial_global_refinement;
      unsigned int max_delta_refinement_level;
      unsigned int mesh_adaptation_frequency;

      BlockSparseMatrix<double> system_matrix;
      BlockSparseMatrix<double> mass_matrix;
      BlockVector<double> system_rhs;
      BlockVector<double> solution;

     

     };


     template <int dim>
     NavierStokes<dim>::NavierStokes()
     : ParameterAcceptor("/NavierStokes/")
     , triangulation(typename Triangulation<dim>::MeshSmoothing(
                   Triangulation<dim>::smoothing_on_refinement |
                   Triangulation<dim>::smoothing_on_coarsening))
      , fe(FE_Q<dim>(2),dim, FE_Q<dim>(1),1)
      , dof_handler(triangulation)
      ,initial_global_refinement(2) 
      ,max_delta_refinement_level(2) 
      , mesh_adaptation_frequency(10)
      ,final_time(1.0)
      ,initial_time(0.0)
      ,time_interval_output(0.1)
      ,abs_tol(1e-6)
      ,rel_tol(1e-6)
      ,is_linear(true)
      ,is_time_ind(true)
      ,max_nonlinear_iterations(10)
      ,max_order_arkode(5)
      ,minimum_step_size(1e-6)
      ,arkode_data()
      {arkode_data.implicit_function_is_linear          = true;  
arkode_data.implicit_function_is_time_independent= true;   
arkode_data.relative_tolerance                   = 1e-6;   
arkode_data.absolute_tolerance                   = 1e-9;
arkode_data.initial_time = 0.0;
arkode_data.final_time = 2.0;


      }     
      
      
    template <int dim>
void NavierStokes<dim>::output_results(const double time,
                                       const unsigned int timestep_number,
                                       const BlockVector<double> &sol)
{
  DataOut<dim> data_out;

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.write_higher_order_cells = true;
  vtk_flags.time  = time;
  vtk_flags.cycle = static_cast<int>(timestep_number);
  data_out.set_flags(vtk_flags);

  std::vector<std::string> names;
  for (unsigned int d=0; d<dim; ++d)
    names.emplace_back(std::string("u_") + (d==0?"x":"y"));
  names.emplace_back("p");

  std::vector<DataComponentInterpretation::DataComponentInterpretation> interp;
  for (unsigned int d=0; d<dim; ++d)
    interp.push_back(DataComponentInterpretation::component_is_part_of_vector);
  interp.push_back(DataComponentInterpretation::component_is_scalar);

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(sol, names,
                           DataOut<dim>::type_dof_data, interp);

  data_out.build_patches(fe.degree); 

  char buf[64];
  std::snprintf(buf, sizeof(buf), "ns_%05u.vtu", timestep_number);
  const std::string vtu_name = buf;

  std::ofstream vtu(vtu_name);
  data_out.write_vtu(vtu);

  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.emplace_back(time, vtu_name);

  std::ofstream pvd("ns_timeseries.pvd");
  DataOutBase::write_pvd_record(pvd, times_and_names);
}
  



    template <int dim>
    void NavierStokes<dim>::setup_system(const double time)
    { 
      dof_handler.distribute_dofs(fe);
      std::cout << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << std::endl;

      hanging_node_constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler,hanging_node_constraints);
      hanging_node_constraints.close();
      


      FEValuesExtractors::Vector velocities(0);
      FEValuesExtractors::Scalar pressure(dim);
    
      auto  boundary_ids =  triangulation.get_boundary_ids();
      homogeneous_constraints.clear();
      homogeneous_constraints.merge(hanging_node_constraints);

      const auto vel_mask = fe.component_mask(velocities);
      std::vector<bool> uy_mask(fe.n_components(), false);
      uy_mask[1]= true;

      Functions::ParsedFunction<dim> vel_exact;

  VectorTools::interpolate_boundary_values(dof_handler, 1,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints, vel_mask);
  VectorTools::interpolate_boundary_values(dof_handler, 3,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints, vel_mask);

  VectorTools::interpolate_boundary_values(dof_handler, 4,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints, uy_mask);

  VectorTools::interpolate_boundary_values(dof_handler, 4,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints,
      fe.component_mask(pressure));

  homogeneous_constraints.close();

  update_current_constraints(time);



      /* for (const auto &boundary_id:boundary_ids)
      { 
        switch (boundary_id)
                {
                  case 1:
                    VectorTools::interpolate_boundary_values(
                      dof_handler,
                      boundary_id,
                      Functions::ZeroFunction<dim>(),
                      homogeneous_constraints,
                      fe.component_mask(velocities));
                    break;
                  case 2:
                    
                    break;
                  case 3:
                    
                      VectorTools::interpolate_boundary_values(
                        dof_handler,
                        boundary_id,
                        Functions::ZeroFunction<dim>(),
                        homogeneous_constraints,
                        fe.component_mask(velocities));
                    break;
                  case 4:
                    VectorTools::interpolate_boundary_values(
                      dof_handler,
                      boundary_id,
                      Functions::ZeroFunction<dim>(),
                      homogeneous_constraints,
                      fe.component_mask(velocities));
                    break;
                  default:
                    AssertThrow(false, ExcMessage("Wrong boundary id"));
                    break;
                }
      } */

      
      std::vector<unsigned int> block_comp(fe.n_components(), 0);
for (unsigned int c = 0; c < dim; ++c) block_comp[c] = 0; 
block_comp[dim] = 1;                                      
DoFRenumbering::component_wise(dof_handler, block_comp);

const auto dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_comp);
const unsigned int n_u = dofs_per_block[0];
const unsigned int n_p = dofs_per_block[1];

solution.reinit(2);
solution.block(0).reinit(n_u);
solution.block(1).reinit(n_p); 
solution.collect_sizes();

system_rhs.reinit(solution);



BlockDynamicSparsityPattern dsp(2,2);
dsp.block(0,0).reinit(n_u, n_u);
dsp.block(0,1).reinit(n_u, n_p); 
dsp.block(1,0).reinit(n_p, n_u); 
dsp.block(1,1).reinit(n_p, n_p); 
dsp.collect_sizes();


DoFTools::make_sparsity_pattern(dof_handler,
                                dsp,
                                homogeneous_constraints,
                                /*keep_constrained_dofs=*/false);


  BlockSparsityPattern bsp;
sparsisty_pattern_.copy_from(dsp);      
    SparsityPattern sp_uu, sp_up, sp_pu;
sp_uu.copy_from(dsp.block(0,0));
sp_up.copy_from(dsp.block(0,1));
sp_pu.copy_from(dsp.block(1,0));

system_matrix.reinit(sparsisty_pattern_);
mass_matrix.reinit(sparsisty_pattern_);

      




      
    }

template <int dim>
void NavierStokes<dim>::setup_arkode(  SUNDIALS::ARKode<BlockVector<double>> &arkode)

{  arkode.implicit_function = [&] (const double t, const BlockVector<double> &y, BlockVector<double> &ydot)
  {
     system_matrix.vmult(ydot,y);
    ydot *= -1.0;
   
    
  };

  arkode.explicit_function = [&](const double t, const BlockVector<double> &y, BlockVector<double> &ydot)
  {
    assemble_convective_rhs_from(y, ydot);
  };
  
  arkode.jacobian_times_vector = [&](const BlockVector<double> &v,
                                    BlockVector<double> &Jv,
                                    double t,
                                    const BlockVector<double> &y,
                                    const BlockVector<double> &fy) {system_matrix.vmult(Jv,v);
                                                                   Jv *= -1.0;};
  

  const auto solve_function =
  [&](SUNDIALS::SundialsOperator<BlockVector<double>>       &op,
        SUNDIALS::SundialsPreconditioner<BlockVector<double>> &prec,
        BlockVector<double>                                   &x,
        const BlockVector<double>                             &b,
        double                                        tol)
        {SolverControl control(1000,1e-6);
         control.log_history(false);
        control.log_result(false);
        SolverGMRES<BlockVector<double>> solver(control);
        solver.solve(op,x,b,prec);
      };

    arkode.solve_linearized_system = solve_function;

   const auto solve_mass_closed = 
     [&](SUNDIALS::SundialsOperator<BlockVector<double>> &,
      SUNDIALS::SundialsPreconditioner<BlockVector<double>> &,
      BlockVector<double> &x, const BlockVector<double> &b, double)
      { x=0.0;
      
       SparseMatrix<double> &Bs=system_matrix.block(1,0);
        SparseMatrix<double> &As = system_matrix.block(0,0);
        SparseMatrix<double> &Bst =system_matrix.block(0,1);
        SparseMatrix<double> &Ms = mass_matrix.block(0,0);
        int n_p= Bs.m();
        int n_u= Ms.m();
        auto M   = linear_operator(mass_matrix.block(0,0));

        SolverControl control(1000,1e-6);
        SolverCG<Vector<double>> cg_solver(control);
       
        dealii::SparseDirectUMFPACK solver;



        cg_solver.solve(mass_matrix.block(0,0), x.block(0), b.block(0), PreconditionIdentity());
        
auto Minv= inverse_operator(M, cg_solver);

auto B   = linear_operator(system_matrix.block(1,0));
auto Bt  = transpose_operator(B);
auto A = linear_operator(system_matrix.block(0,0));
auto S = B * Minv * Bt;
Vector<double> u_rhs(n_u);

 A.vmult(u_rhs,x.block(0)); 
u_rhs +=system_rhs.block(0);

auto B_tmp = B*Minv;
Vector<double> b1(n_p);
B_tmp.vmult(b1,u_rhs);
b1 *= -1.0;
auto Sinv = inverse_operator(S, cg_solver);
S.vmult(x.block(1),b1);


       

      };
  
     
      
    /* arkode.mass_times_vector = [&](const double , 
                                     const BlockVector<double> &v,
                                    BlockVector<double> &Mv){ 
                                      Mv = 0.0;
  mass_matrix.block(0,0).vmult(Mv.block(0), v.block(0));
  Mv.block(1) = 0.0;
                                    };*/
       arkode.solve_mass = solve_mass_closed;
       }
template <int dim>
void NavierStokes<dim>::run()
{
 

  
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    const std::string filename = "nsbench2.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename));
    grid_in.read_ucd(file);
  

  triangulation.refine_global(initial_global_refinement);

 
  setup_system(initial_time);
  assemble_matrix(initial_time);

  
  solution = 0.0;
  current_constraints.distribute(solution);

  
   
  SUNDIALS::ARKode<BlockVector<double>> arkode(arkode_data);
 setup_arkode(arkode);
  
  
  


  double time = 0.0;
  unsigned int step_no = 0;

  output_results(time, step_no, solution);

  
  double next_output_time = time + time_interval_output;

 bool need_reset = true; 

while (time < final_time - 1e-14)
{
  
  if (mesh_adaptation_frequency > 0 && step_no > 0 &&
      (step_no % mesh_adaptation_frequency == 0))
  {
    prepare_for_coarsening_and_refinement(solution);
    std::vector<BlockVector<double>> in{solution}, out;
    transfer_solution_vectors_to_new_mesh(time, in, out);
    solution = out[0];

    assemble_matrix(time);
     
  
  
   

    need_reset = true;
  }

  const double target_time = std::min(next_output_time, final_time);
  std::cout << target_time << std::endl;

  arkode.solve_ode_incrementally(solution, target_time, true); //qui qualcosa non va
  
  
  need_reset = false;

  current_constraints.distribute(solution);
  time = target_time;

  ++step_no;
  output_results(time, step_no, solution);
  next_output_time += time_interval_output;
}


  std::cout << "Finished. Wrote PVD: ns_timeseries.pvd";
}


template <int dim>
void NavierStokes<dim>::assemble_matrix(const double time)
{
  double nu=1.0;
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  QGauss<dim> quadrature(fe.degree+1);

  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_quadrature_points |
                           update_JxW_values | update_gradients);


  const unsigned int n_q_points = quadrature.size();                        

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  system_matrix = 0.0;
  mass_matrix = 0.0;
  system_rhs = 0.0;

  Vector<double> y_mono(dof_handler.n_dofs());
  y_mono = solution;
  
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    std::vector<unsigned int> u_local_pos; u_local_pos.reserve(dofs_per_cell);
    std::vector<unsigned int> p_local_pos; p_local_pos.reserve(dofs_per_cell);

    for (unsigned int i=0; i<dofs_per_cell; i++)
    {
      const unsigned int comp_i = fe.system_to_component_index(i).first;
      if (comp_i < dim)
        u_local_pos.push_back(i);   
      else
        p_local_pos.push_back(i);  
    }
   const unsigned int n_u_cell = u_local_pos.size();
    const unsigned int n_p_cell = p_local_pos.size();

    FullMatrix<double> Mu(n_u_cell, n_u_cell);
    FullMatrix<double> Au(n_u_cell, n_u_cell);
    FullMatrix<double> B (n_p_cell, n_u_cell);
    Vector<double> local_rhs(dofs_per_cell);


   local_rhs = 0.0;
    Mu = 0.0; Au = 0.0; B = 0.0;


    std::vector<Tensor<1,dim>> u_vals(n_q_points);
std::vector<Tensor<2,dim>> grad_u(n_q_points);
fe_values[velocities].get_function_values(y_mono, u_vals);
fe_values[velocities].get_function_gradients(y_mono, grad_u);
     for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double JxW = fe_values.JxW(q);

      
  Tensor<1,dim> conv_q;
  for (unsigned int a = 0; a < dim; ++a)
  {
    double s = 0.0;
    for (unsigned int b = 0; b < dim; ++b)
      s += u_vals[q][b] * grad_u[q][a][b]; 
    conv_q[a] = s;
  }

  for (unsigned int iu = 0; iu < u_local_pos.size(); ++iu)
  {
    const unsigned int i_loc = u_local_pos[iu];
    const Tensor<1,dim> phi_i_u = fe_values[velocities].value(i_loc, q);

    local_rhs(i_loc) += -(conv_q * phi_i_u) * JxW;
  }

      for (unsigned int iu = 0; iu < n_u_cell; ++iu)
      {
        const unsigned int i_loc = u_local_pos[iu];

        const Tensor<1,dim>  phi_i_u   = fe_values[velocities].value(i_loc, q);
        const Tensor<2,dim>  grad_i_u  = fe_values[velocities].gradient(i_loc, q);

        for (unsigned int ju = 0; ju < n_u_cell; ++ju)
        {
          const unsigned int j_loc = u_local_pos[ju];

          const Tensor<1,dim>  phi_j_u  = fe_values[velocities].value(j_loc, q);
          const Tensor<2,dim>  grad_j_u = fe_values[velocities].gradient(j_loc, q);

          Mu(iu, ju) += (phi_i_u * phi_j_u) * JxW;

          Au(iu, ju) += scalar_product(grad_i_u, grad_j_u) * JxW;
        }
      }

      for (unsigned int ip = 0; ip < n_p_cell; ++ip)
      {
        const unsigned int ip_loc = p_local_pos[ip];
        const double       phi_i_p = fe_values[pressure].value(ip_loc, q);

        for (unsigned int ju = 0; ju < n_u_cell; ++ju)
        {
          const unsigned int ju_loc = u_local_pos[ju];
          const double div_phi_j_u  = fe_values[velocities].divergence(ju_loc, q);

         
          B(ip, ju) += (-phi_i_p * div_phi_j_u) * JxW;
        }
      }
    } 
    FullMatrix<double> local_mass_full(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_sys_full (dofs_per_cell, dofs_per_cell);

    local_mass_full = 0.0;
    local_sys_full  = 0.0;

   
    for (unsigned int iu = 0; iu < n_u_cell; ++iu)
    {
      const unsigned int i_loc = u_local_pos[iu];
      for (unsigned int ju = 0; ju < n_u_cell; ++ju)
      {
        const unsigned int j_loc = u_local_pos[ju];

        local_mass_full(i_loc, j_loc) += Mu(iu, ju);          
        local_sys_full (i_loc, j_loc) += nu * Au(iu, ju);     
      }
    }

    
    for (unsigned int ip = 0; ip < n_p_cell; ++ip)
    {
      const unsigned int ip_loc = p_local_pos[ip];
      for (unsigned int ju = 0; ju < n_u_cell; ++ju)
      {
        const unsigned int ju_loc = u_local_pos[ju];

        const double Bij = B(ip, ju);
        local_sys_full(ip_loc, ju_loc) += Bij;  
        local_sys_full(ju_loc, ip_loc) += Bij;  
      }
    }

  
    homogeneous_constraints.distribute_local_to_global(local_mass_full,
                                                       dof_indices,
                                                       mass_matrix);
    homogeneous_constraints.distribute_local_to_global(local_sys_full,
                                                       dof_indices,
                                                       system_matrix);
    current_constraints.distribute_local_to_global(local_rhs, dof_indices, system_rhs);                                  
  } 
}


template <int dim>
void NavierStokes<dim>::assemble_convective_rhs_from(const BlockVector<double> &y,
                                                     BlockVector<double>       &rhs)
{
  rhs = 0.0;

  Vector<double> y_mono(dof_handler.n_dofs());
  y_mono = y;
  current_constraints.distribute(y_mono);

  Vector<double> rhs_mono(dof_handler.n_dofs());
  rhs_mono = 0.0;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature.size();

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    std::vector<unsigned int> u_local_pos; u_local_pos.reserve(dofs_per_cell);
    std::vector<unsigned int> p_local_pos; p_local_pos.reserve(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int comp_i = fe.system_to_component_index(i).first;
      (comp_i < dim ? u_local_pos : p_local_pos).push_back(i);
    }

    Vector<double> local_rhs(dofs_per_cell);
    local_rhs = 0.0;

    std::vector<Tensor<1,dim>> u_vals(n_q_points);
    std::vector<Tensor<2,dim>> grad_u(n_q_points);
    fe_values[velocities].get_function_values   (y_mono, u_vals);
    fe_values[velocities].get_function_gradients(y_mono, grad_u);

    
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double JxW = fe_values.JxW(q);

      Tensor<1,dim> conv_q; 
      for (unsigned int a = 0; a < dim; ++a)
      {
        double s = 0.0;
        for (unsigned int b = 0; b < dim; ++b)
          s += u_vals[q][b] * grad_u[q][a][b]; 
        conv_q[a] = s;
      }

      for (unsigned int iu = 0; iu < u_local_pos.size(); ++iu)
      {
        const unsigned int i_loc = u_local_pos[iu];
        const Tensor<1,dim> phi_i_u = fe_values[velocities].value(i_loc, q);

        local_rhs(i_loc) += -(conv_q * phi_i_u) * JxW; 
      }
    }

    current_constraints.distribute_local_to_global(local_rhs, dof_indices, rhs_mono);
  }

  rhs = rhs_mono;
}



template <int dim>
void NavierStokes<dim>::prepare_for_coarsening_and_refinement( const BlockVector<double> &y)
{
  Vector<double> y_mono(dof_handler.n_dofs());
  y_mono = y;

  const FEValuesExtractors::Vector velocities(0);

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},
                                     y_mono,
                                     estimated_error_per_cell,
                                    fe.component_mask(velocities));
  GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, estimated_error_per_cell,
                                                         0.6,0.4);
  const unsigned int max_grid_level=
                    initial_global_refinement+max_delta_refinement_level;
  const unsigned int min_grid_level = initial_global_refinement;

  if (triangulation.n_levels() > max_grid_level)
     for(const auto &cell:
                        triangulation.active_cell_iterators_on_level(max_grid_level))
                        cell->clear_refine_flag();
      for (const auto &cell :
                        triangulation.active_cell_iterators_on_level(min_grid_level))
                        cell->clear_coarsen_flag();
}

template <int dim>
void NavierStokes<dim>::transfer_solution_vectors_to_new_mesh(
  const double time,
  const std::vector<BlockVector<double>> &all_in,
  std::vector<BlockVector<double>> &all_out)
{ 
  std::vector<Vector<double>> in_mono(all_in.size());
    for (unsigned int i=0;i<all_in.size();++i)
  {
    in_mono[i].reinit(dof_handler.n_dofs());
    in_mono[i] = all_in[i];
  }
  std::vector<const Vector<double>*> in_ptr(all_in.size());
  for (unsigned int i=0;i<all_in.size();++i) in_ptr[i] = &in_mono[i];

  SolutionTransfer<dim,Vector<double>> solution_trans(dof_handler);


 


triangulation.prepare_coarsening_and_refinement();
solution_trans.prepare_for_coarsening_and_refinement(in_mono);
triangulation.execute_coarsening_and_refinement();

setup_system(time);
assemble_matrix(time);

std::vector<Vector<double>> out_mono(all_in.size());
for (auto &v : out_mono) v.reinit(dof_handler.n_dofs());
std::vector<Vector<double>*> out_ptr(all_in.size());
for (unsigned int i=0;i<all_in.size();++i) out_ptr[i] = &out_mono[i];

solution_trans.interpolate(in_mono, out_mono);

 all_out.resize(all_in.size());
  for (unsigned int i=0;i<all_out.size();++i)
  {
    std::vector<unsigned int> block_comp(fe.n_components(), 0);
    for (unsigned int c=0;c<dim;++c) block_comp[c]=0; 
    block_comp[dim]=1;                                
    const auto dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_comp);
    const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];

    all_out[i].reinit(2);
    all_out[i].block(0).reinit(n_u);
    all_out[i].block(1).reinit(n_p);
    all_out[i].collect_sizes();

    all_out[i] = out_mono[i];

    hanging_node_constraints.distribute(all_out[i]);
    current_constraints.distribute(all_out[i]);
  }




}

template <int dim>
void NavierStokes<dim>::update_current_constraints(const double time)
{
  InletVelocity<dim> inlet;
  current_constraints.clear();
  current_constraints.merge(hanging_node_constraints);
  current_constraints.merge(homogeneous_constraints);
  

  VectorTools::interpolate_boundary_values(dof_handler,
                                            2, inlet, current_constraints, fe.component_mask(FEValuesExtractors::Vector(0)));
  current_constraints.close();
                                        
}


}
int main()
{
  try
  {
    using namespace IMEX_NS;
    NavierStokes<2> app;
    app.run();
  }
  catch (std::exception &e)
  {
    std::cerr << "\nException: " << e.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "\nUnknown exception!\n";
    return 1;
  }
  return 0;
}
