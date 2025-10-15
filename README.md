<div style="border: 2px solid #000; padding: 10px; margin-bottom: 20px;">
  <h1 align="center">Machine learning generalised DFT+<em>U</em> projectors</h1>

  <ul style="list-style-position: inside; text-align: center; padding: 0; margin: 10px 0;">
    <li style="margin-bottom: 8px;">
      This repository contains research code for the ChemRxiv manuscript: 
      <strong>Machine learning generalised DFT+<em>U</em> projectors in a numerical atom-centred orbital framework</strong>
    </li>
  </ul>


<div style="border: 2px solid #000; padding: 10px; margin-bottom: 20px;">
  <h1 align="center">Overview</h1>
  
  <p align="center">
    <img src="Overview.png" width="800" />
    <br>
    <em>Tuning Hubbard projectors as a linear combination of numeric atom-centred orbital basis functions, for the simulation of polarons in strongly correlated metal oxides.</em>
  </p>

<div style="border: 2px solid #000; padding: 10px; margin-bottom: 20px;">
  <h1 align="center">Semi-empirical approach</h1>

<ul style="list-style-position: outside; text-align: left; width: 80%; margin: 0 auto; padding-left: 40px;">
  <li>
    <code>Scripts/Semi-empirical Approach/Bayesian.py</code> contains a script to optimise the Ti 3<em>d</em> Hubbard <em>U</em> value and projector for anatase TiO<sub>2</sub> using Bayesian optimisation with a cost function defined using symbolic regression and constraints defined using support vector machines.
  </li>
  
  <p align="center">
    <img src="Semiempirical.png" width="600" />
    <br>
    <em>Integrating symbolic regression, support vector machines and Bayesian optimisation to optimise the Ti 3<em>d</em> Hubbard <em>U</em> value and projectors.</em>
  </p>
</div>

<div style="border: 2px solid #000; padding: 10px; margin-bottom: 20px;">
  <h1 align="center">First-principles approach</h1>

<ul style="list-style-position: outside; text-align: left; width: 80%; margin: 0 auto; padding-left: 40px;">
  <li>
    <code>Scripts/First-principles Approach/Screening/References.txt</code> contains all requirements for screening the the Hubbard <em>U</em> values and projectors of different materials.
  </li>
  <li>
    <code>Scripts/First-principles Approach/Screening/HI-SISSO.py</code> contains hierarchical symbolic regression-defined expressions to calculate DFT+<em>U</em>-predicted orbital occupancies for the materials specified in <code>Scripts/First-principles Approach/Screening/References.txt</code>.
  </li>
  <li>
     <code>Scripts/First-principles Approach/Screening/screen_materials.py</code> uses <code>Scripts/First-principles Approach/Screening/HI-SISSO.py</code> to perform a linear search of the Hubbard parameter space for all materials specified in <code>Scripts/First-principles Approach/Screening/References.txt</code>, whilst evaluating the predicted orbital occupancies using HI-SISSO.py and saving new files {material}_results.txt
  </li>
  <li>
     <code>Scripts/First-principles Approach/Screening/Cost_function.py</code>  uses <code>Scripts/First-principles Approach/Screening/HI-SISSO.py</code> and each {material}_results.txt file to evaluate the first-principles cost function for each screened combination of Hubbard parameters for all materials in <code>Scripts/First-principles Approach/Screening/References.txt</code> and saving new corresponding files {material}_with_JFP.txt
  </li>
  <li>
     <code>Scripts/First-principles Approach/Screening/Familyofsolutions.py</code> uses the outputs from <code>Scripts/First-principles Approach/Screening/screen_materials.py</code> to plot the Hubbard <em>U</em> value and projector coefficients <em>c<sub>1</sub></em> and <em>c<sub>2</sub></em> that minimise the cost function, for each material in <code>Scripts/First-principles Approach/Screening/References.txt</code>.
  </li>
   
  <p align="center">
    <img src="First_principles.png" width="600" />
    <br>
    <em>Developing a first-principles workflow for optimising Hubbard <em>U</em> values and projectors using hierarchical symbolic regression.</em>
  </p>
</div>
