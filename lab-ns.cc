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
  #include <deal.II/sundials/ida.h>
  #include <deal.II/lac/sparse_ilu.h>

  #include <deal.II/lac/petsc_vector.h>
  #include <deal.II/lac/petsc_sparse_matrix.h>
  #include <deal.II/lac/petsc_solver.h>
  #include <deal.II/lac/petsc_precondition.h>
  #include <deal.II/lac/petsc_ts.h>

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
                         const Vector<double> &solution);

      void prepare_for_coarsening_and_refinement(const Vector<double> &solution);
      void transfer_solution_vectors_to_new_mesh(const double time, Vector<double> &y, Vector<double> &y_dot); 
      
      void update_current_constraints(const double time);

      void compute_stokes_initial_guess(Vector<double> &y);

      AffineConstraints<double> hanging_node_constraints;
      AffineConstraints<double> current_constraints;
      AffineConstraints<double> homogeneous_constraints;
      AffineConstraints<double> pressure_boundary_constraints;

      SparsityPattern jac_sparsity_;




      

      SUNDIALS::ARKode<dealii::BlockVector<double>>::AdditionalData arkode_data;


      
      double initial_time;
      double time_interval_output;
      double abs_tol;
      double rel_tol;
      
      bool is_time_ind;
      unsigned int max_nonlinear_iterations;
      unsigned int max_order_arkode;
      double minimum_step_size;
      

      unsigned int initial_global_refinement;
      unsigned int max_delta_refinement_level;

      SparseMatrix<double> system_matrix;
      SparseMatrix<double> mass_matrix;
      Vector<double> system_rhs;
      Vector<double> solution;
      SparseMatrix<double> jacobian_matrix;

      

      void residual(const double time,
                    const Vector<double> &solution,
                             const Vector<double> &solution_dot,
                             Vector<double> &F);
      void assemble_jacobian(const double time,
                                      const Vector<double> &solution,
                                     const Vector<double> &solution_dot,
                                    const double shift);
      
      void solve_with_jacobian( const Vector<double> &src,
                                Vector<double>       &residual,
                                const double tol);
      unsigned int threshold_refine;
      SUNDIALS::IDA<Vector<double>>::AdditionalData data_ida;
      double current_time_for_logs = std::numeric_limits<double>::quiet_NaN();

     

     };


     template <int dim>
     NavierStokes<dim>::NavierStokes()
     : ParameterAcceptor("/NavierStokes/")
     , triangulation(typename Triangulation<dim>::MeshSmoothing(
                   Triangulation<dim>::smoothing_on_refinement |
                   Triangulation<dim>::smoothing_on_coarsening))
      , fe(FE_Q<dim>(2),dim, FE_Q<dim>(1),1)
      , dof_handler(triangulation)
      ,initial_global_refinement(1) 
      ,max_delta_refinement_level(2) 
  
      ,initial_time(0.0)
      ,time_interval_output(0.1)
      ,abs_tol(1e-6)
      ,rel_tol(1e-6)
      ,threshold_refine(1)
   
      ,is_time_ind(true)
      ,max_nonlinear_iterations(10)
      ,max_order_arkode(3)
      ,minimum_step_size(1e-6)
      , data_ida()
      {
       data_ida.initial_time = 0.0;
       data_ida.final_time = 10.0;
       data_ida.initial_step_size = 0.01;
       data_ida.minimum_step_size = 0.001;
       data_ida.absolute_tolerance = 1e-6;
       data_ida.relative_tolerance = 1e-4;
       data_ida.output_period = 0.1;
       data_ida.ic_type = SUNDIALS::IDA<Vector<double>>::AdditionalData::use_y_diff;
       data_ida.reset_type  = SUNDIALS::IDA<Vector<double>>::AdditionalData::use_y_diff;
       data_ida.maximum_non_linear_iterations_ic = 10;
      }     
      
      
    template <int dim>
void NavierStokes<dim>::output_results(const double time,
                                       const unsigned int timestep_number,
                                       const Vector<double> &sol)
{
  DataOut<dim> data_out;

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.write_higher_order_cells = true;
  vtk_flags.time  = time;
  vtk_flags.cycle = static_cast<int>(timestep_number);
  data_out.set_flags(vtk_flags);

  std::vector<std::string> names;
  names = std::vector<std::string>(dim, "u");
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
        std::vector<unsigned int> block_comp(fe.n_components(), 0);
for (unsigned int c = 0; c < dim; ++c) block_comp[c] = 0; 
block_comp[dim] = 1;                                      
DoFRenumbering::component_wise(dof_handler, block_comp);
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
  VectorTools::interpolate_boundary_values(dof_handler, 4,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints, vel_mask);

  

  
IndexSet p_dofs(dof_handler.n_dofs());
p_dofs.add_indices(DoFTools::extract_dofs(dof_handler, fe.component_mask(pressure)));
types::global_dof_index p0 = *p_dofs.begin();  
homogeneous_constraints.add_line(p0);
homogeneous_constraints.set_inhomogeneity(p0, 0.0);
homogeneous_constraints.close();


 



    

      
    

const auto dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_comp);
const unsigned int n_u = dofs_per_block[0];
const unsigned int n_p = dofs_per_block[1];
const unsigned int n_dofs = dof_handler.n_dofs();

DynamicSparsityPattern dsp(n_dofs);
update_current_constraints(time);
DoFTools::make_sparsity_pattern(dof_handler, dsp,current_constraints, /*keep_constrained_dofs=*/true);


jac_sparsity_.copy_from(dsp);
jacobian_matrix.clear();
jacobian_matrix.reinit(jac_sparsity_);

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
 

  SUNDIALS::IDA<Vector<double>> time_stepper(data_ida);
  

time_stepper.residual = [&](const double time,
                                   const Vector<double> &y,
                                   const Vector<double> &y_dot,
                                   Vector<double> &F)
                                   {current_time_for_logs = time;
                                    this->residual(time, y, y_dot, F);};
  time_stepper.setup_jacobian = [&](const double time,
                                const Vector<double> &y,
                               const Vector<double> &y_dot,
                               const double alpha){current_time_for_logs = time;
                                this-> assemble_jacobian(time, y, y_dot, alpha);
                               };
  
  time_stepper.solve_with_jacobian = [&](const  Vector<double> &src,
                                         Vector<double>       &dst,
                                        const double tol) {
                                          std::cout<< current_time_for_logs<<std::endl;
        this->solve_with_jacobian(src, dst,tol);
      };
    time_stepper.output_step = [&](const double t,
                                    const Vector<double> &y,
                                   const Vector<double> &y_dot,
                                  const unsigned int step_number)
                                  {Vector<double> vis = y;
  update_current_constraints(t);
  current_constraints.distribute(vis);    
  std::cout << " step: "<< step_number <<"t= " << t << "  ||y||_inf=" << y.linfty_norm() << std::endl;
  this->output_results(t, step_number, vis); };


    time_stepper.solver_should_restart = [&](const double time,
                                             Vector<double> &y,
                                             Vector<double> &y_dot)
   {
     if(time>= 20*threshold_refine)
     {threshold_refine ++;
      update_current_constraints(time);
          current_constraints.distribute(y);
          std::cout << std::endl << "Adapting the mesh..." << std::endl;
            this->prepare_for_coarsening_and_refinement(y);
            this->transfer_solution_vectors_to_new_mesh(time, y, y_dot);
            return true;


     }
     else
     return false;
                                             };


 time_stepper.differential_components = [&]()
 {  IndexSet diff(dof_handler.n_dofs());
    const FEValuesExtractors::Vector velocities(0);
    diff.add_indices(DoFTools::extract_dofs(dof_handler,fe.component_mask(velocities)));

  
    return diff;};
    
    Vector<double> y(dof_handler.n_dofs()), y_dot(dof_handler.n_dofs());
    compute_stokes_initial_guess(y);
    std::cout<< y.linfty_norm()<<std::endl; 
    




time_stepper.solve_dae(y,y_dot);

  
 
  

  
  
}






template <int dim>
void NavierStokes<dim>::prepare_for_coarsening_and_refinement(const Vector<double> &y)
{
  
  

  const FEValuesExtractors::Vector velocities(0);

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+1),
                                     {},
                                     y,
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
  Vector<double> &y,
Vector<double> &y_dot)
{
 const std::vector< Vector<double>> all_in ={&y,&y_dot};
std::vector<Vector<double>> all_out ={&y,&y_dot};
 SolutionTransfer<dim,Vector<double>> solution_trans(dof_handler);

 
 triangulation.prepare_coarsening_and_refinement();


 solution_trans.prepare_for_coarsening_and_refinement(all_in);
 triangulation.execute_coarsening_and_refinement();
 
 setup_system(time);
 unsigned int n_dofs= dof_handler.n_dofs();
 y.reinit(n_dofs);
 y_dot.reinit(n_dofs);
 solution_trans.interpolate(all_in,all_out); //versione 9.7 è diverso
 


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

template <int dim>
void NavierStokes<dim>::residual(const double time,
                                         const Vector<double> &y,
                                         const Vector<double> &y_dot,
                                         Vector<double> &F )
{
    Vector<double> tmp_solution(y);
   Vector<double> tmp_solution_dot(y_dot);


  update_current_constraints(time);
  current_constraints.distribute(tmp_solution);
  current_constraints.distribute(tmp_solution_dot);

  double nu=0.1;
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  QGauss<dim> quadrature(fe.degree+1);

  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_quadrature_points |
                           update_JxW_values | update_gradients);


  const unsigned int n_q_points = quadrature.size();                        

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  

 Vector<double> cell_residual(dofs_per_cell);
  
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<Tensor<1,dim>> u_vals(n_q_points);
     std::vector<Tensor<1,dim>> u_vals_dot(n_q_points);
std::vector<Tensor<2,dim>> grad_u(n_q_points);
std::vector<double> div_u(n_q_points);
std::vector<double> pres_val(n_q_points);

  F = 0;


  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    cell_residual = 0.0;

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
    
    Mu = 0.0; Au = 0.0; B = 0.0;


fe_values[velocities].get_function_values(tmp_solution, u_vals);
fe_values[velocities].get_function_values(tmp_solution_dot, u_vals_dot);
fe_values[velocities].get_function_gradients(tmp_solution, grad_u);
fe_values[velocities].get_function_divergences(tmp_solution, div_u);
fe_values[pressure].get_function_values(tmp_solution, pres_val);
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

      for (unsigned int iu = 0; iu < n_u_cell; ++iu)
      {
        const unsigned int i_loc = u_local_pos[iu];

        const Tensor<1,dim>  phi_i_u   = fe_values[velocities].value(i_loc, q);
        const Tensor<2,dim>  grad_i_u  = fe_values[velocities].gradient(i_loc, q);
        const double div_phi_i_u  = fe_values[velocities].divergence(i_loc, q);

        
          cell_residual(i_loc)+= (phi_i_u*u_vals_dot[q]+nu*scalar_product(grad_i_u,grad_u[q])+conv_q*phi_i_u-div_phi_i_u*pres_val[q])*JxW;

         
      }

      for (unsigned int ip = 0; ip < n_p_cell; ++ip)
      {
        const unsigned int ip_loc = p_local_pos[ip];
        const double       phi_i_p = fe_values[pressure].value(ip_loc, q);

         
          cell_residual(ip_loc)+= (phi_i_p * div_u[q]) * JxW;
        
      }
    } 
   
current_constraints.distribute_local_to_global(cell_residual,
                                                           local_dof_indices,
                                                           F);
}


for (const auto &c : current_constraints.get_lines())
 {if (c.entries.empty())
   F[c.index]= y[c.index]-tmp_solution[c.index];
  else
   F[c.index]= 0.0;}

}


template <int dim>
void NavierStokes<dim>::assemble_jacobian( const double t,
                                                      const Vector<double> &y,
                                                     const Vector<double> &y_dot,
                                                    const double alpha)
{   jacobian_matrix = 0.0;
   Vector<double> y_mono = y, ydot_mono = y_dot;
  update_current_constraints(t);
  current_constraints.distribute(y_mono);
  current_constraints.distribute(ydot_mono);
    double nu=0.1;
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  QGauss<dim> quadrature(fe.degree+1);

  FEValues<dim> fe_values(fe,
                          quadrature,
                          update_values | update_quadrature_points |
                           update_JxW_values | update_gradients);


  const unsigned int n_q_points = quadrature.size();                        

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  
  

  
  
  
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

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
    


   
    Mu = 0.0; Au = 0.0; B = 0.0;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);


    std::vector<Tensor<1,dim>> u_vals(n_q_points);

    std::vector<Tensor<2,dim>> grad_u(n_q_points);
    fe_values[velocities].get_function_gradients(y_mono,grad_u);
    fe_values[velocities].get_function_values(y_mono,u_vals);

     for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double JxW = fe_values.JxW(q);
      const Tensor<2,dim> &Gu = grad_u[q];


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
          const Tensor<1,dim> u_q = u_vals[q]; 
          double conv_missing = 0.0;
          for (unsigned int a=0; a<dim; ++a)
            for (unsigned int b=0; b<dim; ++b)
                conv_missing += phi_i_u[a] * u_q[b] * grad_j_u[a][b];
          Mu(iu, ju) += (phi_i_u * phi_j_u) * JxW;

          Au(iu, ju) += (scalar_product(grad_i_u, grad_j_u)*nu+ (Gu*phi_j_u)*phi_i_u+conv_missing) * JxW;
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

         
          B(ip, ju) += (phi_i_p * div_phi_j_u) * JxW;
        }
      }
    } 
    FullMatrix<double> local_mass_full(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_sys_full (dofs_per_cell, dofs_per_cell);
   

    cell_matrix = 0.0;

   
    for (unsigned int iu = 0; iu < n_u_cell; ++iu)
    {
      const unsigned int i_loc = u_local_pos[iu];
      for (unsigned int ju = 0; ju < n_u_cell; ++ju)
      {
        const unsigned int j_loc = u_local_pos[ju];

       cell_matrix(i_loc, j_loc) += alpha*Mu(iu, ju)+Au(iu,ju);          
      }
    }

    
    for (unsigned int ip = 0; ip < n_p_cell; ++ip)
    {
      const unsigned int ip_loc = p_local_pos[ip];
      for (unsigned int ju = 0; ju < n_u_cell; ++ju)
      {
        const unsigned int ju_loc = u_local_pos[ju];

        const double Bij = B(ip, ju);
        cell_matrix(ip_loc, ju_loc) += Bij;  
        cell_matrix(ju_loc, ip_loc) -= Bij;  
      }
    }

  
    homogeneous_constraints.distribute_local_to_global(cell_matrix,
                                                       local_dof_indices,
                                                       jacobian_matrix);
   
    
                                     
  }
  
  for (const auto &line : current_constraints.get_lines())
  if (line.entries.empty())          
  {
    const auto i = line.index;
    
    for (auto it = jacobian_matrix.begin(i); it != jacobian_matrix.end(i); ++it)
      jacobian_matrix.set(i, it->column(), 0.0);
    jacobian_matrix.set(i, i, 1.0);
  }
 
}

template <int dim>
void NavierStokes<dim>::solve_with_jacobian( const Vector<double> &src,
                                             Vector<double> &dst,
                                            const double tol)
{
  
    const double rhs_norm = src.l2_norm();
    const double lin_tol = std::max(1e-14,tol*std::max(1.0, rhs_norm));
     SolverControl           solver_control(5000, lin_tol );
     
    try {
     dealii::SparseILU<double> ilu;
dealii::SparseILU<double>::AdditionalData ilu_data;
ilu.initialize(jacobian_matrix, ilu_data);

  SolverGMRES<Vector<double>> gm(solver_control);
  
  gm.solve(jacobian_matrix, dst, src,ilu);
  
} catch (const std::exception &e) {
  std::cerr << "[GMRES] " << e.what() << std::endl;
  dst = 0.0;  
}
  
      std::cout << "     " << solver_control.last_step() << " linear iterations."
            << std::endl;
}


template <int dim>
void NavierStokes<dim>::compute_stokes_initial_guess(Vector<double> &y)
{
  AffineConstraints<double> guess_constraints;
  SparsityPattern sp_guess;
  SparseMatrix<double> laplace_guess;
  Vector<double> y_tmp, rhs_guess;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  InletVelocity<dim> inlet;
  guess_constraints.clear();
  guess_constraints.merge(hanging_node_constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 1,
      Functions::ZeroFunction<dim>(dim+1),
      guess_constraints, fe.component_mask(velocities));
  VectorTools::interpolate_boundary_values(dof_handler, 4,
      Functions::ZeroFunction<dim>(dim+1),
      guess_constraints, fe.component_mask(velocities));
  
  VectorTools::interpolate_boundary_values(dof_handler, 2,
      inlet, guess_constraints, fe.component_mask(velocities));

  
  guess_constraints.close();

 
  DynamicSparsityPattern dsp_guess(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp_guess, guess_constraints);
  sp_guess.copy_from(dsp_guess);
  laplace_guess.reinit(sp_guess);


  y_tmp.reinit(dof_handler.n_dofs()); y_tmp = 0.0;
  rhs_guess.reinit(dof_handler.n_dofs()); rhs_guess = 0.0;

  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
                          update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    std::vector<unsigned int> u_local_pos; u_local_pos.reserve(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      if (fe.system_to_component_index(i).first < dim) u_local_pos.push_back(i);

    cell_matrix = 0.0;

    for (const unsigned int q : fe_values.quadrature_point_indices())
      for (unsigned int iu = 0; iu < u_local_pos.size(); ++iu)
      {
        const unsigned int i_loc = u_local_pos[iu];
        const Tensor<2,dim> grad_i = fe_values[velocities].gradient(i_loc, q);
        for (unsigned int ju = 0; ju < u_local_pos.size(); ++ju) // <-- ++ju
        {
          const unsigned int j_loc = u_local_pos[ju];
          const Tensor<2,dim> grad_j = fe_values[velocities].gradient(j_loc, q);
          cell_matrix(i_loc, j_loc) += scalar_product(grad_i, grad_j) * fe_values.JxW(q);
        }
      }

    guess_constraints.distribute_local_to_global(cell_matrix, local_dof_indices, laplace_guess);
  }

  
  SolverControl solve_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solve_control);
  solver.solve(laplace_guess, y_tmp, rhs_guess, PreconditionIdentity());
  guess_constraints.distribute(y_tmp);

  y = y_tmp; 


  std::vector<std::string> names;
  for (unsigned int d = 0; d < dim; ++d)
    names.push_back(std::string("u_") + (d==0?"x":(d==1?"y":"z")));
  names.push_back("p");

  // interpreta i primi 'dim' come un vettore e l'ultimo come scalare
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(y, names, DataOut<dim>::type_dof_data, interpretation);
  data_out.build_patches(fe.degree); // o un numero fisso (ad es. 2) se preferisci

  std::ofstream out("stokes_guess.vtu");
  data_out.write_vtu(out);
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







/*
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
          SolverControl control(1000,1e-6);
        SolverCG<Vector<double>> cg_solver(control);
        cg_solver.solve(mass_matrix.block(0,0), x.block(0), b.block(0), PreconditionIdentity()); 

        
      
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
        

        Vector<double> xcop;
        xcop.reinit(n_u);
        cg_solver.solve(mass_matrix.block(0,0), xcop, b.block(0), PreconditionIdentity());
        x.block(0)=xcop;
        
auto Minv= inverse_operator(M, cg_solver);

auto B   = linear_operator(system_matrix.block(1,0));
auto Bt  = transpose_operator(B);
auto A = linear_operator(system_matrix.block(0,0));
auto S = B * Minv * Bt;
Vector<double> u_rhs(n_u);

 A.vmult(u_rhs,xcop); 
u_rhs +=system_rhs.block(0);

auto B_tmp = B*Minv;
Vector<double> b1(n_p);
B_tmp.vmult(b1,u_rhs);
b1 *= -1.0;

auto Sinv = inverse_operator(S, cg_solver);

Sinv.vmult(x.block(1),b1);
std::cout<<"re"<< std::endl;


       

      };
  
     
      
    
       arkode.solve_mass = solve_mass_closed; 
       arkode.mass_times_vector = [&](const double , 
                                     const BlockVector<double> &v,
                                    BlockVector<double> &Mv){ 
                                     mass_matrix.vmult(Mv,v);
                                    
                                    };
       } 
                                    
       
       template <int dim>
void NavierStokes<dim>::assemble_convective_rhs_from(const Vector<double> &y,
                                                     Vector<double>       &rhs)
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


  template <int dim>
void NavierStokes<dim>::explicit_function(const double time, const PETScWrappers::VectorBase &y, 
                             PETScWrappers::VectorBase &rhs)
{
    rhs = 0.0;

  PETScWrappers::VectorBase tmp_solution;

  update_current_constraints(time);
  current_constraints.distribute(tmp_solution);
  

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_gradients | update_JxW_values|update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature.size();

  Vector<double> cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
   std::vector<Tensor<1,dim>> u_vals(n_q_points);
std::vector<Tensor<2,dim>> grad_u(n_q_points);
std::vector<double> div_u(n_q_points);

rhs=0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    std::vector<unsigned int> u_local_pos; u_local_pos.reserve(dofs_per_cell);
    std::vector<unsigned int> p_local_pos; p_local_pos.reserve(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int comp_i = fe.system_to_component_index(i).first;
      if (comp_i < dim)
        u_local_pos.push_back(i);   
      else
        p_local_pos.push_back(i); 
    }
    const unsigned int n_u_cell = u_local_pos.size();
    const unsigned int n_p_cell = p_local_pos.size();

    Vector<double> local_rhs(dofs_per_cell);
    local_rhs = 0.0;

  
    fe_values[velocities].get_function_values   (tmp_solution, u_vals);
    fe_values[velocities].get_function_gradients(tmp_solution, grad_u);
    

    
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

    current_constraints.distribute_local_to_global(local_rhs, local_dof_indices, rhs);
  }
rhs.compress(VectorOperation::add);

for (const auto &c : current_constraints.get_lines())
 {if (c.entries.empty())
   rhs[c.index]= y[c.index]-tmp_solution[c.index];
  else
   rhs[c.index]= y[c.index];}
rhs.compress(VectorOperation::insert);
}
}



std::vector<PETScWrappers::VectorBase> all_in_ghosted(all_in.size());
std::vector<PETScWrappers::VectorBase *> all_in_ghosted_ptr(all_in.size());
std::vector<PETScWrappers::VectorBase *> all_out_ptr(all_in.size());
for (unsigned int i=0;i<all_in.size();i++)
{
  all_in_ghosted[i].reinit();
  all_in_ghosted[i] = all_in[i];
  all_in_ghosted_otr[i] = &all_in_ghosted[i];
}

triangulation.prepare_coarsening_and_refinement();
solution_trans.prepare_for_coarsening_and_refinement(all_in_ghosted_ptr);
triangulation.execute_coarsening_and_refinement();

setup_system(time);

all_out.resize(all_in.size());
for(unsigned int i=0;i<all_in.size(); i++)
{
  all_out[i].reinit();
  all_out_ptr[i] = &all_out[i];
}
solution_trans.interpolate(all_out_ptr);

for (PETScWrappers::VectorBase &v : all_out)
   hanging_node_constraints.distribute(v);*/