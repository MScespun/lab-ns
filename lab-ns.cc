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
  #include <deal.II/base/function.h>

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

#include <iostream>
#include <fstream>
#include <iomanip>


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
     unsigned int fe_degree_vel;
     unsigned int fe_degree_pre;
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

      double initial_time;
      double final_time;
      double initial_step_size;
      double output_period;
      double minimum_step_size;
      unsigned int maximum_order;
      
      double abs_tol;
      double rel_tol;
      unsigned int max_non_linear_it_ic;
      
      
      

      unsigned int initial_global_refinement;
      unsigned int max_delta_refinement_level;
      

      double nu;
      
      double ref_time;

      std::vector<std::string> exact_solution_expression;
      std::vector<std::string> forcing_expression;
      std::vector<std::string> exact_derivative_expression;
      FunctionParser<dim> exact_solution;
      FunctionParser<dim> forcing_function;
      FunctionParser<dim> exact_derivative;

      std::ofstream out;



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

      AffineConstraints<double> derivative_constraints;

      void update_derivative_constraints(double time);
      void project_div_free(Vector<double> &y);
      void rebuild_parsed_functions();

     

     };


     template <int dim>
     NavierStokes<dim>::NavierStokes()
     : ParameterAcceptor("/NavierStokes/")
      ,triangulation(typename Triangulation<dim>::MeshSmoothing(
                   Triangulation<dim>::smoothing_on_refinement |
                   Triangulation<dim>::smoothing_on_coarsening))
      , fe_degree_vel(2)
      , fe_degree_pre(1)
      , fe(FE_Q<dim>(fe_degree_vel),dim, FE_Q<dim>(fe_degree_pre),1)
      
      , dof_handler(triangulation) 
      , initial_time(0.0)
      ,final_time(25.0)
      , initial_step_size(0.01)
      ,output_period(0.1)
      ,minimum_step_size(1e-6)
      , maximum_order(5)
      ,abs_tol(1e-6)
      ,rel_tol(1e-6)
      ,max_non_linear_it_ic(20)
      ,initial_global_refinement(1)
      , max_delta_refinement_level(2)
      ,nu(0.01)
      ,ref_time(1.0)
      ,exact_solution_expression({"sin(x)*cos(y)*exp(-2*nu*t)","-cos(x)*sin(y)*exp(-2*nu*t)","0.25*(cos(2*x)+cos(2*y))*exp(-4*nu*t)"})
      , forcing_expression({"0","0"})
      , exact_solution(dim+1)
      ,exact_derivative_expression({"-2*nu*sin(x)*cos(y)*exp(-2*nu*t)","2*nu*cos(x)*sin(y)*exp(-2*nu*t)","-nu*(cos(2*x)+cos(2*y))*exp(-4*nu*t)"})
      , forcing_function(dim)
      ,exact_derivative(dim+1)
      , out("dati.csv")
      ,threshold_refine(1)
      , data_ida()
      {
        enter_subsection("Timestepper parameters");
      {add_parameter("Initial time", initial_time);
      add_parameter("Final time", final_time);
       add_parameter("Initial step size", initial_step_size);
       add_parameter("Output period", output_period);
       add_parameter("Minimum step size", minimum_step_size);
       add_parameter("Maximum order", maximum_order);

       add_parameter("Absolute tolerance", abs_tol);
       add_parameter("Relative tolerance", rel_tol);
       add_parameter("Maximum non linear iterations for IC", max_non_linear_it_ic);
      }
      leave_subsection();
      enter_subsection("Mesh parameters");
      {
       add_parameter("Initial global refinement", initial_global_refinement);
       add_parameter("Maximum delta refinement level", max_delta_refinement_level);
       add_parameter("Finite element degree vel", fe_degree_vel);
       add_parameter("Finite element degree pres", fe_degree_pre);
      }
      leave_subsection();

      enter_subsection("Test");
      {add_parameter("Viscosity", nu);
      add_parameter("Exact solution x", exact_solution_expression[0]);
      
      add_parameter("Exact solution y", exact_solution_expression[1]);
      add_parameter("Exact solution p", exact_solution_expression[2]);
      add_parameter("Forcing term x", forcing_expression[0]);
      add_parameter("Forcing term y", forcing_expression[1]);
      add_parameter("Exact derivative x", exact_derivative_expression[0]);
      
      add_parameter("Exact derivative y", exact_derivative_expression[1]);
      add_parameter("Exact derivative p", exact_derivative_expression[2]);
      
  }
  leave_subsection();
      
      std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;
    constants["lambda"]=0.5*(1/nu-sqrt(1/(std::pow(nu,2))+16*std::pow(numbers::PI,2)));
       exact_solution.initialize(FunctionParser<dim>::default_variable_names()+",t",
                              exact_solution_expression,
                              constants, true);
    forcing_function.initialize(FunctionParser<dim>::default_variable_names(),
                            forcing_expression,
                            constants);
         exact_derivative.initialize(FunctionParser<dim>::default_variable_names()+",t",
                              exact_derivative_expression,
                              constants, true);
      

    
      
      }     
  
template <int dim>
void NavierStokes<dim>::update_derivative_constraints(double time)
{
  derivative_constraints.clear();
  derivative_constraints.merge(hanging_node_constraints);
  exact_derivative.set_time(time);
  const FEValuesExtractors::Vector velocities(0);
  const auto vel_mask = fe.component_mask(velocities);

  VectorTools::interpolate_boundary_values(dof_handler, 0,
      exact_derivative, derivative_constraints);



  

  derivative_constraints.close();
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
  std::snprintf(buf, sizeof(buf), "ns_test%05u.vtu", timestep_number);
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
      const FEValuesExtractors::Scalar v_y(1); 
    
      auto  boundary_ids =  triangulation.get_boundary_ids();
      homogeneous_constraints.clear();
      homogeneous_constraints.merge(hanging_node_constraints);

      const auto vel_mask = fe.component_mask(velocities);
      std::vector<bool> uy_mask(fe.n_components(), false);
      uy_mask[1]= true;

      Functions::ParsedFunction<dim> vel_exact;

      VectorTools::interpolate_boundary_values(dof_handler, 0,
      exact_solution, homogeneous_constraints);
      
/* 
  VectorTools::interpolate_boundary_values(dof_handler, 1,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints, vel_mask);
  VectorTools::interpolate_boundary_values(dof_handler, 4,
      Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints, vel_mask);
      
  VectorTools::interpolate_boundary_values(dof_handler,3, Functions::ZeroFunction<dim>(dim+1), homogeneous_constraints,fe.component_mask(v_y));
  VectorTools::interpolate_boundary_values(
  dof_handler,
  3,
  Functions::ZeroFunction<dim>(dim+1),
  homogeneous_constraints,
  fe.component_mask(pressure));
  */

  

  
homogeneous_constraints.close();
update_current_constraints(0);

 



    

      
    

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
{ out << "Time, Vel_err, Vel_rel_err, Pres_err, Pres_rel_err \n";
  data_ida.initial_time = initial_time;
       data_ida.final_time = final_time;
       data_ida.initial_step_size = initial_step_size;
       data_ida.minimum_step_size = minimum_step_size;
       data_ida.absolute_tolerance = abs_tol;
       data_ida.relative_tolerance = rel_tol;
       data_ida.output_period = output_period;
       data_ida.ic_type = SUNDIALS::IDA<Vector<double>>::AdditionalData::use_y_diff;
       data_ida.reset_type  = SUNDIALS::IDA<Vector<double>>::AdditionalData::use_y_diff;
       data_ida.maximum_non_linear_iterations_ic = max_non_linear_it_ic;
       data_ida.maximum_order = maximum_order;
  rebuild_parsed_functions();
  std::cout<<"test";
 GridGenerator::hyper_cube(triangulation,0.0, 2*numbers::PI );

  /* 
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    const std::string filename = "nsbench2.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename));
    grid_in.read_ucd(file);*/
  

  triangulation.refine_global(initial_global_refinement);

 
  setup_system(initial_time);
 

  SUNDIALS::IDA<Vector<double>> time_stepper(data_ida);
  

time_stepper.residual = [&](const double time,
                                   const Vector<double> &y,
                                   const Vector<double> &y_dot,
                                   Vector<double> &F)
                                   {
                                    update_current_constraints(time);
                                    this->residual(time, y, y_dot, F);};
  time_stepper.setup_jacobian = [&](const double time,
                                const Vector<double> &y,
                               const Vector<double> &y_dot,
                               const double alpha){
                                update_current_constraints(time);
                                this-> assemble_jacobian(time, y, y_dot, alpha);
                                 //std::cout<< "setup_jacobian"<<std::endl;
                               };
  
  time_stepper.solve_with_jacobian = [&](const  Vector<double> &src,
                                         Vector<double>       &dst,
                                        const double tol) {
                                         
        this->solve_with_jacobian(src, dst,tol);
      };
    time_stepper.output_step = [&](const double t,
                                    const Vector<double> &y,
                                   const Vector<double> &y_dot,
                                  const unsigned int step_number)
                                  {Vector<double> vis = y;
  update_current_constraints(t);
  homogeneous_constraints.distribute(vis);    
    exact_solution.set_time(t);
    const FEValuesExtractors::Vector velocities(0);
const FEValuesExtractors::Scalar pressure(dim);
const ComponentMask vel_mask = fe.component_mask(velocities);
const ComponentMask prs_mask = fe.component_mask(pressure);

auto vel_selected= DoFTools::extract_dofs(dof_handler, vel_mask);
auto prs_selected = DoFTools::extract_dofs(dof_handler, prs_mask);
std::vector<types::global_dof_index> vel_dofs, prs_dofs;
vel_dofs.reserve(dof_handler.n_dofs());
prs_dofs.reserve(dof_handler.n_dofs());

const ComponentSelectFunction<dim> vel_sel(std::make_pair(0, dim),
                                                       dim + 1);     
Vector<double> zero(dof_handler.n_dofs());
zero = 0.0;
const ComponentSelectFunction<dim> prs_sel(dim,dim+1);      
  Vector<double> err_vel(vel_dofs.size());
  Vector<double> err_pres(prs_dofs.size());
  Vector<double> norm_v_exact(vel_dofs.size());
Vector<double> norm_p_exact(prs_dofs.size());
  VectorTools::integrate_difference(dof_handler, vis, exact_solution, err_vel, QGauss<dim>(fe.degree+1), VectorTools::H1_norm, &vel_sel);
  VectorTools::integrate_difference(dof_handler, vis, exact_solution, err_pres, QGauss<dim>(fe.degree+1), VectorTools::L2_norm, &prs_sel);
  VectorTools::integrate_difference(dof_handler,
                                  zero,
                                  exact_solution,
                                  norm_v_exact,
                                  QGauss<dim>(fe.degree+1),
                                  VectorTools::H1_norm,
                                  &vel_sel);

VectorTools::integrate_difference(dof_handler,
                                  zero,
                                  exact_solution,
                                  norm_p_exact,
                                  QGauss<dim>(fe.degree+1),
                                  VectorTools::L2_norm,
                                  &prs_sel);
  const double err_vel_compute = VectorTools::compute_global_error(triangulation, err_vel,  VectorTools::H1_norm);
  const double err_pres_compute = VectorTools::compute_global_error(triangulation, err_pres, VectorTools::L2_norm);
    const double norm_v_exact_compute= VectorTools::compute_global_error(triangulation, norm_v_exact,  VectorTools::H1_norm);
  const double norm_p_exact_compute = VectorTools::compute_global_error(triangulation, norm_p_exact, VectorTools::L2_norm);
  std::cout<< t<< std::endl;
  out << std::fixed << std::setprecision(6)
            << t << "," << err_vel_compute <<","<<  err_vel_compute/norm_v_exact_compute<<","<< err_pres_compute <<"," << err_pres_compute/norm_p_exact_compute <<"\n";
  

  this->output_results(t, step_number, vis); };


    time_stepper.solver_should_restart = [&](const double time,
                                             Vector<double> &y,
                                             Vector<double> &y_dot)
   { 
     if(time>= 50*threshold_refine)
     {threshold_refine ++; 
          
     Vector<double> tmp_y = y;
     Vector<double> tmp_y_dot = y_dot;
     
   
     
          std::cout << std::endl << "Adapting the mesh..." << std::endl;
            this->prepare_for_coarsening_and_refinement(y);
            this->transfer_solution_vectors_to_new_mesh(time, tmp_y, tmp_y_dot);
            y= tmp_y;
            y_dot = tmp_y_dot;
            y_dot=0.0;

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
    //compute_stokes_initial_guess(y);
    
   // y_dot=0.0;
  //  update_current_constraints(initial_time);
//update_derivative_constraints(initial_time);



Vector<double> y_exact(dof_handler.n_dofs());
Vector<double> ydot_exact(dof_handler.n_dofs());
exact_solution.set_time(initial_time);
exact_derivative.set_time(initial_time);

VectorTools::interpolate(dof_handler, exact_solution, y_exact);
VectorTools::interpolate(dof_handler, exact_derivative, ydot_exact);
//current_constraints.distribute(y_exact);                // su y
//derivative_constraints.distribute(ydot_exact);



time_stepper.solve_dae(y_exact,ydot_exact);

  
 
  

  
  
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
  update_current_constraints(time);           
  Vector<double> y_in = y;
  current_constraints.distribute(y_in);        

  const std::vector<Vector<double>> all_in = {y_in};
  SolutionTransfer<dim, Vector<double>> soltrans(dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  soltrans.prepare_for_coarsening_and_refinement(all_in);

  triangulation.execute_coarsening_and_refinement();

  setup_system(time);                         

  std::vector<Vector<double>> all_out(1);
  all_out[0].reinit(dof_handler.n_dofs());

  soltrans.interpolate(all_in, all_out);

  y.reinit(dof_handler.n_dofs());
  y      = all_out[0];
  update_current_constraints(time);           
  current_constraints.distribute(y);          
  y_dot.reinit(dof_handler.n_dofs());
  y_dot = 0.0;
   project_div_free(y);






}


template <int dim>
void NavierStokes<dim>::update_current_constraints(const double time)
{
  InletVelocity<dim> inlet;
  current_constraints.clear();
  current_constraints.merge(hanging_node_constraints);
  exact_solution.set_time(time);
  //std::cout<< "time: " <<time << "sol: "<< exact_solution.value(Point<dim>(numbers::PI,numbers::PI),2)<<std::endl;
  const FEValuesExtractors::Vector velocities(0);
  const auto vel_mask = fe.component_mask(velocities);
  VectorTools::interpolate_boundary_values(dof_handler, 0,
      exact_solution, current_constraints);
  

  //VectorTools::interpolate_boundary_values(dof_handler,
   //                                         2, inlet, current_constraints, fe.component_mask(FEValuesExtractors::Vector(0)));
  current_constraints.close();
  homogeneous_constraints.clear();
  homogeneous_constraints.merge(current_constraints);
  homogeneous_constraints.close();
                                        
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
  update_derivative_constraints(time);
  current_constraints.distribute(tmp_solution);
  derivative_constraints.distribute(tmp_solution_dot);

  
  
  

  
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

  F = 0.0;


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

  
    current_constraints.distribute_local_to_global(cell_matrix,
                                                       local_dof_indices,
                                                       jacobian_matrix);
   
    
                                     
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
SparseDirectUMFPACK direct;
  direct.initialize(jacobian_matrix);
  direct.vmult(dst, src);        
  

//  SolverGMRES<Vector<double>> gm(solver_control);
  
  //gm.solve(jacobian_matrix, dst, src,ilu);
  
} catch (const std::exception &e) {
  std::cerr << "[GMRES] " << e.what() << std::endl;
  dst = 0.0;  
}
}


template <int dim>
void NavierStokes<dim>::compute_stokes_initial_guess(Vector<double> &y)
{FEValuesExtractors::Vector velocities(0);
  
  AffineConstraints<double> guess_constraints;
  guess_constraints.clear();
  
  guess_constraints.merge(hanging_node_constraints); 
  exact_solution.set_time(0.0);
  
  const auto vel_mask = fe.component_mask(velocities);
  VectorTools::interpolate_boundary_values(dof_handler, 0,
      exact_solution, guess_constraints, fe.component_mask(FEValuesExtractors::Vector(0)));
  
 
  //VectorTools::interpolate_boundary_values(dof_handler,
   //                                         2, inlet, current_constraints, fe.component_mask(FEValuesExtractors::Vector(0)));

  guess_constraints.close();

 
  SparsityPattern sp;
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, guess_constraints,
                                    true);
    sp.copy_from(dsp);
  }
  SparseMatrix<double> stokes_matrix;
  stokes_matrix.reinit(sp);

  Vector<double> rhs, y_tmp;
  rhs.reinit(dof_handler.n_dofs());
  rhs = 0.0;
  y_tmp.reinit(dof_handler.n_dofs());
  y_tmp = 0.0;

 
  
  const FEValuesExtractors::Scalar pressure(dim);
  QGauss<dim> quad(fe.degree + 1);
  FEValues<dim> fe_values(fe, quad,
                          update_values | update_gradients |
                          update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    Vector<double> cell_rhs(dofs_per_cell);
    cell_rhs = 0.0;
    std::vector<unsigned int> u_pos; u_pos.reserve(dofs_per_cell);
    std::vector<unsigned int> p_pos; p_pos.reserve(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int comp = fe.system_to_component_index(i).first;
      (comp < dim ? u_pos : p_pos).push_back(i);
    }
    const unsigned int n_u = u_pos.size();
    const unsigned int n_p = p_pos.size();

    FullMatrix<double> Au(n_u, n_u);
    FullMatrix<double> B (n_p, n_u);
    Au = 0.0; B = 0.0;

    for (unsigned int q = 0; q < quad.size(); ++q)
    {
      const double JxW = fe_values.JxW(q);

      for (unsigned int iu = 0; iu < n_u; ++iu)
      {
        const unsigned int i_loc = u_pos[iu];
        const Tensor<2,dim> grad_i_u = fe_values[velocities].gradient(i_loc, q);

        for (unsigned int ju = 0; ju < n_u; ++ju)
        {
          const unsigned int j_loc = u_pos[ju];
          const Tensor<2,dim> grad_j_u = fe_values[velocities].gradient(j_loc, q);
          Au(iu, ju) += nu * scalar_product(grad_i_u, grad_j_u) * JxW;
        }
      }

      for (unsigned int ip = 0; ip < n_p; ++ip)
      {
        const unsigned int ip_loc = p_pos[ip];
        const double phi_i_p = fe_values[pressure].value(ip_loc, q);

        for (unsigned int ju = 0; ju < n_u; ++ju)
        {
          const unsigned int j_loc = u_pos[ju];
          const double div_phi_j_u = fe_values[velocities].divergence(j_loc, q);
          B(ip, ju) += (phi_i_p * div_phi_j_u) * JxW; 
        }
      }
    }

    FullMatrix<double> cell_mat(dofs_per_cell, dofs_per_cell);
    cell_mat = 0.0;

    
    for (unsigned int iu = 0; iu < n_u; ++iu)
      for (unsigned int ju = 0; ju < n_u; ++ju)
        cell_mat(u_pos[iu], u_pos[ju]) += Au(iu, ju);

    
    for (unsigned int ip = 0; ip < n_p; ++ip)
      for (unsigned int ju = 0; ju < n_u; ++ju)
      {
        const double Bij = B(ip, ju);
        cell_mat(p_pos[ip], u_pos[ju]) +=  Bij; 
        cell_mat(u_pos[ju], p_pos[ip]) += -Bij; 
      }

    
    guess_constraints.distribute_local_to_global(cell_mat,
      cell_rhs,
                                                 local_dof_indices,
                                                stokes_matrix,rhs);
  }


  SparseDirectUMFPACK direct;
  direct.initialize(stokes_matrix);
  direct.vmult(y_tmp, rhs);        
  guess_constraints.distribute(y_tmp);


  y = y_tmp;

 
}
template <int dim>
void NavierStokes<dim>::project_div_free(Vector<double> &y)
{
  // y = [u;p]. 
  Vector<double> y_rhs = y;
  AffineConstraints<double> guess_constraints;
   guess_constraints.clear();

  guess_constraints.merge(homogeneous_constraints); 

const FEValuesExtractors::Vector velocities(0);
/*  InletVelocity<dim> inlet;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           2,
                                           inlet,
                                           guess_constraints,
                                           fe.component_mask(velocities));*/
  guess_constraints.close();

  
  
  SparsityPattern sp;
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, current_constraints, /*keep_constrained=*/true);
    sp.copy_from(dsp);
  }
  SparseMatrix<double> K;   
  K.reinit(sp);
  Vector<double> rhs(dof_handler.n_dofs());
  rhs = 0.0;


  const FEValuesExtractors::Scalar pressure(dim);
  QGauss<dim> quad(fe.degree+1);
  FEValues<dim> fe_values(fe, quad, update_values|update_gradients|update_JxW_values|update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(dof_indices);

    std::vector<unsigned int> u_pos, p_pos;
    u_pos.reserve(dofs_per_cell); p_pos.reserve(dofs_per_cell);
    for (unsigned int i=0;i<dofs_per_cell;++i)
      (fe.system_to_component_index(i).first < dim ? u_pos : p_pos).push_back(i);

    const unsigned int n_u = u_pos.size(), n_p = p_pos.size();
    FullMatrix<double> Muu(n_u,n_u);     Muu = 0.0;
    FullMatrix<double> Bpu(n_p,n_u);     Bpu = 0.0;
    Vector<double>     fu(dofs_per_cell); fu  = 0.0;

    std::vector<Tensor<1,dim>> u_interp_vals(quad.size());
    fe_values[velocities].get_function_values(y_rhs, u_interp_vals);

    for (unsigned int q=0;q<quad.size();++q)
    {
      const double JxW = fe_values.JxW(q);

  
      for (unsigned int iu=0; iu<n_u; ++iu)
      {
        const auto i_loc = u_pos[iu];
        const Tensor<2,dim> grad_i_u = fe_values[velocities].gradient(i_loc,q);
        const Tensor<1,dim> phi_i = fe_values[velocities].value(i_loc,q);

        for (unsigned int ju=0; ju<n_u; ++ju)
        {
          const auto j_loc = u_pos[ju];
          const Tensor<2,dim> grad_j_u = fe_values[velocities].gradient(j_loc,q);
          const Tensor<1,dim> phi_j = fe_values[velocities].value(j_loc,q);
          Muu(iu,ju) += (phi_i * phi_j) * JxW  +  nu * scalar_product(grad_i_u, grad_j_u) * JxW;;
        }

      
        fu(i_loc) += (phi_i * u_interp_vals[q]) * JxW;
      }

   
      for (unsigned int ip=0; ip<n_p; ++ip)
      {
        const auto p_loc = p_pos[ip];
        const double qi = fe_values[pressure].value(p_loc,q);

        for (unsigned int ju=0; ju<n_u; ++ju)
        {
          const auto j_loc = u_pos[ju];
          const double div_phi_j = fe_values[velocities].divergence(j_loc,q);
          Bpu(ip,ju) += qi * div_phi_j * JxW;
        }
      }
    }

    
    FullMatrix<double> cell_mat(dofs_per_cell, dofs_per_cell);
    cell_mat = 0.0;

    for (unsigned int iu=0; iu<n_u; ++iu)
      for (unsigned int ju=0; ju<n_u; ++ju)
        cell_mat(u_pos[iu], u_pos[ju]) += Muu(iu,ju);

    for (unsigned int ip=0; ip<n_p; ++ip)
      for (unsigned int ju=0; ju<n_u; ++ju)
      {
        const double Bij = Bpu(ip,ju);
        cell_mat(p_pos[ip], u_pos[ju]) +=  Bij; 
        cell_mat(u_pos[ju], p_pos[ip]) +=  -Bij;
      }

    Vector<double> cell_rhs(dofs_per_cell);
    cell_rhs = 0.0;
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      cell_rhs(i) += fu(i);

    guess_constraints.distribute_local_to_global(cell_mat, cell_rhs, dof_indices, K, rhs);
  }

  
  SparseDirectUMFPACK direct;
  direct.initialize(K);
  Vector<double> y_corr(dof_handler.n_dofs());
  direct.vmult(y_corr, rhs);

  current_constraints.distribute(y_corr);
  y = y_corr; 
}

template <int dim>
void NavierStokes<dim>::rebuild_parsed_functions() {
  std::map<std::string,double> constants;
  constants["pi"] = numbers::PI;
  constants["lambda"] =
      0.5*(1.0/nu - std::sqrt(1.0/(nu*nu) + 16.0*numbers::PI*numbers::PI));
  constants["nu"] = nu;
      
  exact_solution.initialize(FunctionParser<dim>::default_variable_names()+",t",
                            exact_solution_expression, constants, true);
  forcing_function.initialize(FunctionParser<dim>::default_variable_names(),
                              forcing_expression, constants);
    exact_derivative.initialize(FunctionParser<dim>::default_variable_names()+",t",
                              exact_derivative_expression,
                              constants, true);
}


}
int main()
{
  try
  {
    
    using namespace IMEX_NS;
   
    NavierStokes<2> app;
    
    ParameterAcceptor::initialize("parameter-file.prm");
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
