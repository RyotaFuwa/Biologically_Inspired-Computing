import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.util.*;
import java.lang.*;

import static org.epochx.stats.StatField.*;

import org.epochx.gp.model.*;
import org.epochx.gp.op.init.*;
import org.epochx.gp.op.crossover.*;
import org.epochx.gp.op.mutation.*;
import org.epochx.op.selection.*;
import org.epochx.life.*;
import org.epochx.stats.*;
import org.epochx.tools.random.MersenneTwisterFast;

import org.epochx.epox.*;
import org.epochx.epox.math.*;
import org.epochx.epox.lang.*;
import org.epochx.gp.representation.*;
import org.epochx.representation.CandidateProgram;
import org.epochx.tools.util.BoolUtils;
import org.epochx.tools.random.MersenneTwisterFast;

public class GPwrapper extends GPModel  
{
	private int numAtts;
	private InstanceSet is;
	private Variable[] vars;
	private int numConstants = 0;

	private int maxDepth 			= getMaxDepth();
	private int populationSize		= 250;
	private int noGenerationNum     = 50;
	private int noEliteNum   		= 10;
	private int TournamentNum 		= 10;
	private double crossoverProb  	= 0.2;
	private double mutationProb 	= 0.2;
	private double reprocutionProb  = 0.2;

	public GPwrapper(InstanceSet pIs) {
		numAtts=Attributes.getNumAttributes();
		is=pIs;
	}

	public void parseParams(String paramFile) {
		try {
			Properties properties = new Properties();
			properties.load(new FileInputStream(paramFile));

			Enumeration enuKeys = properties.keys();
			while (enuKeys.hasMoreElements()) {
				String key = (String) enuKeys.nextElement();
				String value = properties.getProperty(key);
	
				if(key.equals("populationSize")) {
					populationSize = Integer.parseInt(value);
				} else if(key.equals("noGenerationNum")) {
					noGenerationNum = Integer.parseInt(value);
				} else if(key.equals("maxDepth")) {
					maxDepth = Integer.parseInt(value);
				} else if(key.equals("noEliteNum")) {
					noEliteNum = Integer.parseInt(value);
				} else if(key.equals("TournamentNum")) {
					TournamentNum = Integer.parseInt(value);
				} else if(key.equals("mutationProb")) {
					mutationProb = Double.parseDouble(value);
				} else if(key.equals("crossoverProb")) {
					crossoverProb = Double.parseDouble(value);
				} else if(key.equals("reproductionProb")) {
					reprocutionProb = Double.parseDouble(value);
				} else if(key.equals("numConstants")) {
					numConstants = Integer.parseInt(value);
				} else {
					System.out.println("Unknown parameter "+key);
				}  
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public double getFitness(CandidateProgram p) {
		GPCandidateProgram program = (GPCandidateProgram) p;

		ClassifierGP cGP = new ClassifierGP(program, vars);

		Instance[] instances = is.getInstances();
		int diff = 0;
		int errorNum = 0;
		for (int i=0; i<instances.length; i++) {
			/*
			int lossValue = cGP.calcFitness1(instances[i]);
			int pred = lossValue<0?0:1;
			// System.out.println(lossValue);
			if(pred != instances[i].getClassValue()){
				errorNum += (int)Math.max(0, Math.abs(lossValue));
			}
			*/
			int pred = cGP.classifyInstance(instances[i]);
			if(pred != instances[i].getClassValue()){
				errorNum++;
			}
		}
	   	// return errorNum + program.getProgramLength();
	   	// return Math.pow(errorNum, 1.5) + program.getProgramLength();
	   	return Math.pow(errorNum, 3);
	}

	@Override
	public Class<?> getReturnType() {
		return Double.class;
	}

	public Classifier generateClassifier() {
		parseParams("config/gp.config");
		vars = new Variable[numAtts];
		double min=0,max=0;
		for(int i=0;i<numAtts;i++) {
			Attribute att = Attributes.getAttribute(i);
			if(i==0) {
				min = att.minAttribute();
				max = att.maxAttribute();
			} else {
				if(att.minAttribute()<min) {
					min = att.minAttribute();
				}
				if(att.maxAttribute()>max) {
					max = att.maxAttribute();
				}
			}
			vars[i] = new Variable("var"+i,Double.class);
		}

		List<Node> syntax = new ArrayList<Node>();
		// Terminals. First, the placeholders for the data attributes
		for(int i=0;i<numAtts;i++) {
			syntax.add(vars[i]);
		}

		// Functions.
		syntax.add(new AddFunction());
		syntax.add(new MultiplyFunction());
		syntax.add(new DivisionProtectedFunction());
		syntax.add(new SubtractFunction());

	    // syntax.add(new AbsoluteFunction());
		syntax.add(new CubeFunction());
		// syntax.add(new CubeRootFunction());
		// syntax.add(new ExponentialFunction());
		// syntax.add(new FactorialFunction());
		// syntax.add(new GreaterThanFunction());
		syntax.add(new InvertProtectedFunction());
		// syntax.add(new LessThanFunction());
		// syntax.add(new Log10Function());
		// syntax.add(new LogFunction());
		// syntax.add(new Max2Function());
		// syntax.add(new Max3Function());
		// syntax.add(new Min2Function());
		// syntax.add(new Min3Function());
		// syntax.add(new ModuloProtectedFunction());
		syntax.add(new PowerFunction());
		// syntax.add(new SquareFunction());
	

		// And now the random constants, set within the domain of the data variable
		MersenneTwisterFast prng = new MersenneTwisterFast();
		DoubleERC generator = new DoubleERC(prng,min,max,5);
		for(int i=0;i<numConstants;i++) {
			syntax.add(generator.newInstance());
		}

		// double constant = min;
		// double step = (max-min)/(double)(numConstants-1);
		// for(int i=0;i<numConstants;i++) {
			// Literal lit = new Literal(constant);
			// syntax.add(lit);
			// constant+=step;
		// }

		setSyntax(syntax);

		// Set parameters.
		setMaxDepth(maxDepth);
		setPopulationSize(populationSize);
		setNoGenerations(noGenerationNum);
		setNoElites(noEliteNum);
		setCrossoverProbability(crossoverProb);
		setMutationProbability(mutationProb);
		setReproductionProbability(reprocutionProb);

		// Set operators and components.
		setInitialiser(new RampedHalfAndHalfInitialiser (this));
		// setInitialiser(new GrowInitialiser(this));
		// setMutation(new PointMutation(this));
		setProgramSelector(new TournamentSelector(this, TournamentNum));
		// setCrossover(new KozaCrossover(this));
		setMaxInitialDepth(10);


		Life.get().addGenerationListener(new GenerationAdapter(){
			public void onGenerationEnd() {
					Stats.get().print(GEN_NUMBER, GEN_FITNESS_MIN);
			}
		});

		// Run the model.
		run();
		GPCandidateProgram bestProgram = (GPCandidateProgram) Stats.get().getStat(RUN_FITTEST_PROGRAM);
		// System.out.println("Best program:");
		System.out.println(bestProgram);

		return new ClassifierGP(bestProgram,vars);
	}
}
