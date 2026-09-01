import { Match, Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as cdk from '../../core';
import * as codebuild from '../lib';

test('can attach permissions boundary to Project', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  const project = new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHub({ owner: 'a', repo: 'b' }),
  });
  iam.PermissionsBoundary.of(project).apply(new codebuild.UntrustedCodeBoundaryPolicy(stack, 'Boundary'));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
    PermissionsBoundary: { Ref: 'BoundaryEA298153' },
  });
});

test('can add additional statements Boundary', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  const project = new codebuild.Project(stack, 'Project', {
    source: codebuild.Source.gitHub({ owner: 'a', repo: 'b' }),
  });
  iam.PermissionsBoundary.of(project).apply(new codebuild.UntrustedCodeBoundaryPolicy(stack, 'Boundary', {
    additionalStatements: [
      new iam.PolicyStatement({
        actions: ['a:a'],
        resources: ['arn:aws:s3:::my-bucket/my-object'],
      }),
    ],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::ManagedPolicy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Effect: 'Allow',
        Action: 'a:a',
        Resource: 'arn:aws:s3:::my-bucket/my-object',
      }]),
    },
  });
});
