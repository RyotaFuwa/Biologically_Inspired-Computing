/**
 * Class that implements a GP classifier
 */

import org.epochx.gp.representation.GPCandidateProgram;
import org.epochx.epox.*;

public class ClassifierGP extends Classifier implements Cloneable {
	
	GPCandidateProgram program;
	Variable[] vars;

	public ClassifierGP(GPCandidateProgram pProgram, Variable[] pVars) {
		program =pProgram;
		vars = pVars;
	}

	static public double sigmoid(double x){
		return 1 / ( 1 + Math.exp(-x));
	}

	public int calcFitness1(Instance ins) {
		for(int j=0;j<vars.length;j++) {
			vars[j].setValue(ins.getAttribute(j));
		}

		Double result = (Double) program.evaluate();
		return result.intValue();
		/*
		for(int j=0;j<vars.length;j++) {
			vars[j].setValue(ins.getAttribute(j));
		}

		Double result = (Double) program.evaluate();
		return result.doubleValue();
		*/
	}

	public int classifyInstance(Instance ins) {
		for(int j=0;j<vars.length;j++) {
			vars[j].setValue(ins.getAttribute(j));
		}

		Double result = (Double) program.evaluate();
		return result<0?0:1;
	}
	
	public void printClassifier() {
		System.out.println(program);
	}

	public int classifierClass() {
		return 0;
	}

}
