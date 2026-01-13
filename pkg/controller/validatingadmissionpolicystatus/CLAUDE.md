# Package: validatingadmissionpolicystatus

## Purpose
The ValidatingAdmissionPolicy Status controller reconciles the Status field of ValidatingAdmissionPolicy objects by running type checks against the policy's CEL expressions and updating the status with any warnings.

## Key Types

- **Controller**: Main controller that watches ValidatingAdmissionPolicy objects and updates their status with type checking results.

## Key Functions

- **NewController(policyInformer, policyClient, typeChecker)**: Creates a new controller instance.
- **Run(ctx, workers)**: Starts the controller workers.
- **enqueuePolicy(policy)**: Adds a policy to the work queue for processing.
- **reconcile(ctx, policy)**: Performs type checking on the policy and updates its status via server-side apply.

## Design Notes

- Controller name is "validatingadmissionpolicy-status" to differentiate from the API server's admission controller.
- Uses server-side apply (ApplyStatus) with force=true to update policy status.
- Only processes policies where Generation > Status.ObservedGeneration.
- Type checking is performed by an external TypeChecker that validates CEL expressions against the policy's ParamsKind and MatchConstraints.
- Expression warnings are stored in Status.TypeChecking.ExpressionWarnings.
