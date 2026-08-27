import type { Resource, Service, SpecDatabase } from '@aws-cdk/service-spec-types';
import { emptyDatabase } from '@aws-cdk/service-spec-types';
import type { Plain } from '@cdklabs/tskb';
import { TypeScriptRenderer } from '@cdklabs/typewriter';
import { moduleForResource } from './util';
import { AwsCdkLibBuilder } from '../lib/cdk/aws-cdk-lib';
import './expect-to-contain-code';

const renderer = new TypeScriptRenderer();
let db: SpecDatabase;
let service: Service;

const BASE_RESOURCE: Plain<Resource> = {
  name: 'Resource',
  primaryIdentifier: ['Id'],
  attributes: {},
  properties: {
    Id: {
      type: { type: 'string' },
      documentation: 'The identifier of the resource',
    },
  },
  cloudFormationType: 'AWS::Some::Resource',
};

beforeEach(async () => {
  db = emptyDatabase();

  service = db.allocate('service', {
    name: 'aws-some',
    shortName: 'some',
    capitalized: 'Some',
    cloudFormationNamespace: 'AWS::Some',
  });
});

test('resource interface when cfnRefIdentifier is a property', () => {
  // GIVEN
  const resource = db.allocate('resource', BASE_RESOURCE);
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource with arnTemplate', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    ...BASE_RESOURCE,
    arnTemplate: 'arn:${Partition}:some:${Region}:${Account}:resource/${ResourceId}',
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource with optional primary identifier gets property from ref', () => {
  // GIVEN
  const resource = db.allocate('resource', BASE_RESOURCE);
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource with multiple primaryIdentifiers as properties', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    ...BASE_RESOURCE,
    primaryIdentifier: ['Id', 'AnotherId'],
    properties: {
      Id: {
        type: { type: 'string' },
        documentation: 'The identifier of the resource',
      },
      AnotherId: {
        type: { type: 'string' },
        documentation: 'Another identifier of the resource',
      },
    },
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource interface when primaryIdentifier is an attribute', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    ...BASE_RESOURCE,
    primaryIdentifier: ['Id'],
    properties: {},
    attributes: {
      Id: {
        type: { type: 'string' },
        documentation: 'The identifier of the resource',
      },
    },
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource interface with multiple primaryIdentifiers', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    ...BASE_RESOURCE,
    primaryIdentifier: ['Id', 'Another'],
    properties: {},
    attributes: {
      Id: {
        type: { type: 'string' },
        documentation: 'The identifier of the resource',
      },
      Another: {
        type: { type: 'string' },
        documentation: 'Another identifier of the resource',
      },
    },
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource interface with "Arn"', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    name: 'Resource',
    primaryIdentifier: ['Id'],
    properties: {},
    attributes: {
      Id: {
        type: { type: 'string' },
        documentation: 'The identifier of the resource',
      },
      Arn: {
        type: { type: 'string' },
        documentation: 'The arn for the resource',
      },
    },
    cloudFormationType: 'AWS::Some::Resource',
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource interface with "<Resource>Arn"', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    name: 'Something',
    primaryIdentifier: ['Id'],
    properties: {},
    attributes: {
      Id: {
        type: { type: 'string' },
        documentation: 'The identifier of the resource',
      },
      SomethingArn: {
        type: { type: 'string' },
        documentation: 'The arn for something',
      },
    },
    cloudFormationType: 'AWS::Some::Something',
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Something').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('an unrecorded attribute name conflict fails codegen', () => {
  // GIVEN
  givenResource({
    ...BASE_RESOURCE,
    attributes: {
      'FooBar': {
        type: { type: 'string' },
        documentation: 'The FooBar of the resource',
      },
      'Foo.Bar': {
        type: { type: 'string' },
        documentation: 'The Foo.Bar of the resource',
      },
    },
  });

  // THEN
  expect(() => renderResource()).toThrow(
    "Attribute name conflict on AWS::Some::Resource between 'FooBar' and 'Foo.Bar', which both become 'attrFooBar'. Add an entry in attribute-name-conflict-resolutions.ts",
  );
});

test('resource interface with Arn as a property and not a primaryIdentifier', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    name: 'Resource',
    primaryIdentifier: ['Id'],
    properties: {
      Arn: {
        type: { type: 'string' },
        documentation: 'The arn for the resource',
      },
    },
    attributes: {
      Id: {
        type: { type: 'string' },
        documentation: 'The identifier of the resource',
      },
    },
    cloudFormationType: 'AWS::Some::Resource',
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('resource interface with Arn as primaryIdentifier', () => {
  // GIVEN
  const resource = db.allocate('resource', {
    name: 'Resource',
    primaryIdentifier: ['Arn'],
    properties: {},
    attributes: {
      Arn: {
        type: { type: 'string' },
        documentation: 'The arn of the resource',
      },
    },
    cloudFormationType: 'AWS::Some::Resource',
  });
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const module = moduleForResource(foundResource, { db });

  const rendered = renderer.render(module);

  expect(rendered).toMatchSnapshot();
});

test('can generate interface types into a separate module', () => {
  // GIVEN
  const resource = db.allocate('resource', BASE_RESOURCE);
  db.link('hasResource', service, resource);

  // THEN
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', 'AWS::Some::Resource').only();

  const ast = new AwsCdkLibBuilder({ db });
  const info = ast.addResource(foundResource);
  const rendered = {
    interfaces: renderer.render(info.interfaces.module),
    resources: renderer.render(info.resourcesMod.module),
  };

  expect(rendered.interfaces).toMatchSnapshot();
  expect(rendered.resources).toMatchSnapshot();
});

test('reference interface is based on CC-API identifier with values pulled from attributes', () => {
  // GIVEN
  givenResource({
    ...BASE_RESOURCE,
    attributes: {
      ParentId: {
        type: { type: 'string' },
        documentation: 'The identifier of the parent resource',
      },
    },
    primaryIdentifier: ['ParentId', 'Id'],
    cfnRefIdentifier: ['Id'],
  });

  // THEN
  const rendered = renderResource();
  expect(rendered.interfaces).toContainCode(
    `export interface ResourceReference {
      /**
       * The ParentId of the Resource resource.
       */
      readonly parentId: string;

      /**
       * The Id of the Resource resource.
       */
      readonly resourceId: string;
    }`);
  expect(rendered.resources).toContainCode(
    `public get resourceRef(): ResourceReference {
      return {
        parentId: this.attrParentId,
        resourceId: this.ref
      };
    }`,
  );
});

test('reference interface is based on CC-API identifier with values pulled required properties', () => {
  // GIVEN
  givenResource({
    ...BASE_RESOURCE,
    properties: {
      ...BASE_RESOURCE.properties,
      ParentId: {
        type: { type: 'string' },
        documentation: 'The identifier of the parent resource',
        required: true,
      },
    },
    primaryIdentifier: ['ParentId', 'Id'],
    cfnRefIdentifier: ['Id'],
  });

  // THEN
  const rendered = renderResource();
  expect(rendered.interfaces).toContainCode(
    `export interface ResourceReference {
      /**
       * The ParentId of the Resource resource.
       */
      readonly parentId: string;

      /**
       * The Id of the Resource resource.
       */
      readonly resourceId: string;
    }`);
  expect(rendered.resources).toContainCode(
    `public get resourceRef(): ResourceReference {
      return {
        parentId: this.parentId,
        resourceId: this.ref
      };
    }`,
  );
});

test('optional properties are dropped from CC-API-based reference interface', () => {
  // GIVEN
  givenResource({
    ...BASE_RESOURCE,
    properties: {
      ...BASE_RESOURCE.properties,
      ParentId: {
        type: { type: 'string' },
        documentation: 'The identifier of the parent resource',
      },
    },
    primaryIdentifier: ['ParentId', 'Id'],
    cfnRefIdentifier: ['Id'],
  });

  // THEN
  const rendered = renderResource();
  expect(rendered.interfaces).toContainCode(
    `export interface ResourceReference {
      /**
       * The Id of the Resource resource.
       */
      readonly resourceId: string;
    }`);
  expect(rendered.resources).toContainCode(
    `public get resourceRef(): ResourceReference {
      return {
        resourceId: this.ref
      };
    }`,
  );
});

test('CFN reference identifier of same length as CC-API identifier aliases field names', () => {
  // GIVEN
  givenResource({
    ...BASE_RESOURCE,
    attributes: {
      Arn: {
        type: { type: 'string' },
        documentation: 'The ARN of this resource',
      },
    },
    primaryIdentifier: ['Id'],
    cfnRefIdentifier: ['Arn'],
  });

  // THEN
  const rendered = renderResource();
  expect(rendered.interfaces).toContainCode(
    `export interface ResourceReference {
      /**
       * The Arn of the Resource resource.
       */
      readonly resourceArn: string;
    }`);
  expect(rendered.resources).toContainCode(
    `public get resourceRef(): ResourceReference {
      return {
        resourceArn: this.ref
      };
    }`,
  );
});

function givenResource(res: Plain<Resource>) {
  db.link('hasResource', service, db.allocate('resource', res));
}

function renderResource(resourceType = 'AWS::Some::Resource') {
  const foundResource = db.lookup('resource', 'cloudFormationType', 'equals', resourceType).only();

  const ast = new AwsCdkLibBuilder({ db });
  const info = ast.addResource(foundResource);
  return {
    interfaces: renderer.render(info.interfaces.module),
    resources: renderer.render(info.resourcesMod.module),
  };
}
