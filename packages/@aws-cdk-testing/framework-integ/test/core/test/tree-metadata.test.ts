/**
 * Unit tests that depend on 'aws-cdk-lib' having been compiled using jsii
 */
import * as fs from 'fs';
import * as path from 'path';
import { Construct } from 'constructs';
import * as cxschema from 'aws-cdk-lib/cloud-assembly-schema';
import type { TreeInspector } from 'aws-cdk-lib';
import { App, AssumptionError, CfnParameter, CfnResource, Lazy, Stack, Validations, aws_s3 } from 'aws-cdk-lib';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type { ForestFile, TreeFile } from 'aws-cdk-lib/core/lib/helpers-internal';

abstract class AbstractCfnResource extends CfnResource {
  constructor(scope: Construct, id: string) {
    super(scope, id, {
      type: 'CDK::UnitTest::MyCfnResource',
    });
  }

  public inspect(inspector: TreeInspector) {
    inspector.addAttribute('aws:cdk:cloudformation:type', 'CDK::UnitTest::MyCfnResource');
    inspector.addAttribute('aws:cdk:cloudformation:props', this.cfnProperties);
  }

  protected abstract override get cfnProperties(): { [key: string]: any };
}

let app: App;
beforeEach(() => {
  app = new App();
  acknowledgeProblems();
});

function acknowledgeProblems() {
  Validations.of(app).acknowledge(
    { id: 'CloudFormation-Validate::F3006', reason: 'We are using fake resource types here' },
    { id: 'CloudFormation-Validate::F0001', reason: 'Empty resource sections' },
  );
}

describe('tree metadata', () => {
  test('tree metadata is generated as expected', () => {
    const stack = new Stack(app, 'mystack');
    new Construct(stack, 'myconstruct');

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    expect(readJson(assembly.directory, treeArtifact!.file)).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        id: 'App',
        path: '',
        children: {
          Tree: expect.objectContaining({
            id: 'Tree',
            path: 'Tree',
          }),
          mystack: expect.objectContaining({
            id: 'mystack',
            path: 'mystack',
            children: {
              BootstrapVersion: {
                constructInfo: {
                  fqn: expect.stringMatching(/(aws-cdk-lib.CfnParameter|aws-cdk-lib.CfnElement|constructs.Construct)/),
                  version: expect.any(String),
                },
                id: 'BootstrapVersion',
                path: 'mystack/BootstrapVersion',
              },
              CheckBootstrapVersion: {
                constructInfo: {
                  fqn: expect.stringMatching(/(aws-cdk-lib.CfnRule|constructs.Construct)/),
                  version: expect.any(String),
                },
                id: 'CheckBootstrapVersion',
                path: 'mystack/CheckBootstrapVersion',
              },
              myconstruct: expect.objectContaining({
                id: 'myconstruct',
                path: 'mystack/myconstruct',
              }),
            },
          }),
        },
      }),
    });
  });

  test('L1 resource add logicalID to tree metadata', () => {
    const stack = new Stack(app, 'mystack');
    new aws_s3.Bucket(stack, 'Bucky');

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    expect(readJson(assembly.directory, treeArtifact!.file)).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        children: expect.objectContaining({
          mystack: expect.objectContaining({
            id: 'mystack',
            path: 'mystack',
            children: expect.objectContaining({
              Bucky: expect.objectContaining({
                id: 'Bucky',
                path: 'mystack/Bucky',
                children: expect.objectContaining({
                  Resource: expect.objectContaining({
                    id: 'Resource',
                    path: 'mystack/Bucky/Resource',
                    attributes: expect.objectContaining({
                      'aws:cdk:cloudformation:logicalId': 'Bucky3D4A6827',
                      'aws:cdk:cloudformation:type': 'AWS::S3::Bucket',
                    }),
                  }),
                }),
              }),
            }),
          }),
        }),
      }),
    });
  });

  test('tree metadata for a Cfn resource', () => {
    class MyCfnResource extends AbstractCfnResource {
      protected get cfnProperties(): { [key: string]: any } {
        return {
          mystringpropkey: 'mystringpropval',
          mylistpropkey: ['listitem1'],
          mystructpropkey: {
            myboolpropkey: true,
            mynumpropkey: 50,
          },
        };
      }
    }

    const stack = new Stack(app, 'mystack');
    new MyCfnResource(stack, 'mycfnresource');

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    expect(readJson(assembly.directory, treeArtifact!.file)).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        id: 'App',
        path: '',
        children: {
          Tree: expect.objectContaining({
            id: 'Tree',
            path: 'Tree',
          }),
          mystack: expect.objectContaining({
            id: 'mystack',
            path: 'mystack',
            children: expect.objectContaining({
              mycfnresource: expect.objectContaining({
                id: 'mycfnresource',
                path: 'mystack/mycfnresource',
                attributes: {
                  'aws:cdk:cloudformation:type': 'CDK::UnitTest::MyCfnResource',
                  'aws:cdk:cloudformation:props': {
                    mystringpropkey: 'mystringpropval',
                    mylistpropkey: ['listitem1'],
                    mystructpropkey: {
                      myboolpropkey: true,
                      mynumpropkey: 50,
                    },
                  },
                },
              }),
            }),
          }),
        },
      }),
    });
  });

  test('tree metadata has construct class & version in there', () => {
    // The runtime metadata this test relies on is only available if the most
    // recent compile has happened using 'jsii', as the jsii compiler injects
    // this metadata.
    //
    // If the most recent compile was using 'tsc', the metadata will not have
    // been injected, and the test will fail.
    //
    // People may choose to run `tsc` directly (instead of `yarn build` for
    // example) to escape the additional TSC compilation time that is necessary
    // to run 'eslint', or the additional time that 'jsii' needs to analyze the
    // type system), this test is allowed to fail if we're not running on CI.
    //
    // If the compile of this library has been done using `tsc`, the runtime
    // information will always find `constructs.Construct` as the construct
    // identifier, since `constructs` will have had a release build done using `jsii`.
    //
    // If this test is running on CodeBuild, we will require that the more specific
    // class names are found. If this test is NOT running on CodeBuild, we will
    // allow the specific class name (for a 'jsii' build) or the generic
    // 'constructs.Construct' class name (for a 'tsc' build).
    const stack = new Stack(app, 'mystack');
    new CfnResource(stack, 'myconstruct', { type: 'Aws::Some::Resource' });

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    const treeJson = readJson(assembly.directory, treeArtifact!.file);

    expect(treeJson).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        children: expect.objectContaining({
          mystack: expect.objectContaining({
            constructInfo: {
              fqn: expect.stringMatching(/\bStack$/),
              version: expect.any(String),
            },
            children: expect.objectContaining({
              myconstruct: expect.objectContaining({
                constructInfo: {
                  fqn: expect.stringMatching(/\bCfnResource$/),
                  version: expect.any(String),
                },
              }),
            }),
          }),
        }),
      }),
    });
  });

  /**
   * Check that we can limit ourselves to a given tree file size
   *
   * We can't try the full 512MB because the test process will run out of memory
   * before synthing such a large tree.
   */
  test('tree.json can be split over multiple files', () => {
    const MAX_NODES = 1_000;
    app = new App({
      context: {
        '@aws-cdk/core.TreeMetadata:maxNodes': MAX_NODES,
      },
      analyticsReporting: false,
    });
    acknowledgeProblems();

    // GIVEN
    const buildStart = Date.now();
    const addedNodes = recurseBuild(app, 4, 4);

    console.log('Built tree in', Date.now() - buildStart, 'ms');

    // WHEN
    const synthStart = Date.now();
    const assembly = app.synth();

    console.log('Synthed tree in', Date.now() - synthStart, 'ms');
    try {
      const treeArtifact = assembly.tree();
      expect(treeArtifact).toBeDefined();

      // THEN - does not explode, and file sizes are correctly limited
      const treeSizes: Record<string, number> = {};
      recurseVisit(assembly.directory, treeArtifact!.file, undefined, treeSizes);

      // `treeSizes` has keys of the form `<file>#<tree id>`. Combine them.
      const fileSizes: Record<string, number> = {};
      for (const [treeName, count] of Object.entries(treeSizes)) {
        const fileName = treeName.split('#')[0];
        if (fileName in fileSizes) {
          fileSizes[fileName] += count;
        } else {
          fileSizes[fileName ] = count;
        }
      }

      for (const size of Object.values(treeSizes)) {
        expect(size).toBeLessThanOrEqual(MAX_NODES);
      }
      for (const size of Object.values(fileSizes)) {
        expect(size).toBeLessThanOrEqual(MAX_NODES);
      }

      expect(Object.keys(treeSizes).length).toBeGreaterThan(1);
      expect(Object.keys(fileSizes).length).toBeLessThan(Object.keys(treeSizes).length);

      const foundNodes = sum(Object.values(treeSizes));
      expect(foundNodes).toEqual(addedNodes + 2); // App, Tree
    } finally {
      fs.rmSync(assembly.directory, { force: true, recursive: true });
    }

    function recurseBuild(scope: Construct, n: number, depth: number) {
      if (depth === 0) {
        const resourceCount = 450;
        const stack = new Stack(scope, 'SomeStack');
        for (let i = 0; i < resourceCount; i++) {
          new CfnResource(stack, `Resource${i}`, { type: 'Aws::Some::Resource' });
        }
        return resourceCount + 3; // Also count Stack, BootstrapVersion, CheckBootstrapVersion
      }

      let ret = 0;
      for (let i = 0; i < n; i++) {
        const parent = new Construct(scope, `Construct${i}`);
        ret += 1;
        ret += recurseBuild(parent, n, depth - 1);
      }
      return ret;
    }

    function recurseVisit(directory: string, fileName: string, treeId: string | undefined, files: Record<string, number>) {
      let nodes: number;

      if (treeId) {
        // Assume a forest file
        const treeJson: ForestFile = readJson(directory, fileName);
        for (const [tid, tree] of Object.entries(treeJson.forest)) {
          nodes = 0;
          rec(tree);
          files[`${fileName}#${tid}`] = nodes;
        }
      } else {
        // Assume a tree file
        const treeJson: TreeFile = readJson(directory, fileName);
        nodes = 0;
        rec(treeJson.tree);
        files[fileName] = nodes;
      }

      function rec(x: TreeFile['tree']) {
        if (isSubtreeReference(x)) {
          // We'll count this node as part of our visit to the "real" node
          recurseVisit(directory, x.fileName, x.treeId, files);
        } else {
          nodes += 1;
          for (const child of Object.values(x.children ?? {})) {
            rec(child);
          }
        }
      }
    }
  });

  test('token resolution & cfn parameter', () => {
    const stack = new Stack(app, 'mystack');
    const cfnparam = new CfnParameter(stack, 'mycfnparam');

    class MyCfnResource extends AbstractCfnResource {
      protected get cfnProperties(): { [key: string]: any } {
        return {
          lazykey: Lazy.string({ produce: () => 'LazyResolved!' }),
          cfnparamkey: cfnparam,
        };
      }
    }

    new MyCfnResource(stack, 'mycfnresource');

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    expect(readJson(assembly.directory, treeArtifact!.file)).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        id: 'App',
        path: '',
        children: {
          Tree: expect.objectContaining({
            id: 'Tree',
            path: 'Tree',
          }),
          mystack: expect.objectContaining({
            id: 'mystack',
            path: 'mystack',
            children: expect.objectContaining({
              mycfnparam: expect.objectContaining({
                id: 'mycfnparam',
                path: 'mystack/mycfnparam',
              }),
              mycfnresource: expect.objectContaining({
                id: 'mycfnresource',
                path: 'mystack/mycfnresource',
                attributes: {
                  'aws:cdk:cloudformation:type': 'CDK::UnitTest::MyCfnResource',
                  'aws:cdk:cloudformation:props': {
                    lazykey: 'LazyResolved!',
                    cfnparamkey: { Ref: 'mycfnparam' },
                  },
                },
              }),
            }),
          }),
        },
      }),
    });
  });

  test('cross-stack tokens', () => {
    class MyFirstResource extends AbstractCfnResource {
      public readonly lazykey: string;

      constructor(scope: Construct, id: string) {
        super(scope, id);
        this.lazykey = Lazy.string({ produce: () => 'LazyResolved!' });
      }

      protected get cfnProperties(): { [key: string]: any } {
        return {
          lazykey: this.lazykey,
        };
      }
    }

    class MySecondResource extends AbstractCfnResource {
      public readonly myprop: string;

      constructor(scope: Construct, id: string, myprop: string) {
        super(scope, id);
        this.myprop = myprop;
      }

      protected get cfnProperties(): { [key: string]: any } {
        return {
          myprop: this.myprop,
        };
      }
    }

    const firststack = new Stack(app, 'myfirststack');
    const firstres = new MyFirstResource(firststack, 'myfirstresource');
    const secondstack = new Stack(app, 'mysecondstack');
    new MySecondResource(secondstack, 'mysecondresource', firstres.lazykey);

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    expect(readJson(assembly.directory, treeArtifact!.file)).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        id: 'App',
        path: '',
        children: {
          Tree: expect.objectContaining({
            id: 'Tree',
            path: 'Tree',
          }),
          myfirststack: expect.objectContaining({
            id: 'myfirststack',
            path: 'myfirststack',
            children: expect.objectContaining({
              myfirstresource: expect.objectContaining({
                id: 'myfirstresource',
                path: 'myfirststack/myfirstresource',
                attributes: {
                  'aws:cdk:cloudformation:type': 'CDK::UnitTest::MyCfnResource',
                  'aws:cdk:cloudformation:props': {
                    lazykey: 'LazyResolved!',
                  },
                },
              }),
            }),
          }),
          mysecondstack: expect.objectContaining({
            id: 'mysecondstack',
            path: 'mysecondstack',
            children: expect.objectContaining({
              mysecondresource: expect.objectContaining({
                id: 'mysecondresource',
                path: 'mysecondstack/mysecondresource',
                attributes: {
                  'aws:cdk:cloudformation:type': 'CDK::UnitTest::MyCfnResource',
                  'aws:cdk:cloudformation:props': {
                    myprop: 'LazyResolved!',
                  },
                },
              }),
            }),
          }),
        },
      }),
    });
  });

  test('tree metadata can be disabled', () => {
    // GIVEN
    app = new App({
      treeMetadata: false,
    });
    acknowledgeProblems();

    // WHEN
    const stack = new Stack(app, 'mystack');
    new Construct(stack, 'myconstruct');

    const assembly = app.synth();
    const treeArtifact = assembly.tree();

    // THEN
    expect(treeArtifact).not.toBeDefined();
  });

  test('failing nodes', () => {
    class MyCfnResource extends CfnResource {
      public inspect(_: TreeInspector) {
        throw new AssumptionError(lit`ForcedInspectError`, 'Forcing an inspect error');
      }
    }

    const stack = new Stack(app, 'mystack');
    new MyCfnResource(stack, 'mycfnresource', {
      type: 'CDK::UnitTest::MyCfnResource',
    });

    const assembly = app.synth();
    const treeArtifact = assembly.tree();
    expect(treeArtifact).toBeDefined();

    const treenode = app.node.findChild('Tree');

    const warn = treenode.node.metadata.find((md) => {
      return md.type === cxschema.ArtifactMetadataEntryType.WARN
        && /Forcing an inspect error/.test(md.data as string)
        && /mycfnresource/.test(md.data as string);
    });
    expect(warn).toBeDefined();

    // assert that the rest of the construct tree is rendered
    expect(readJson(assembly.directory, treeArtifact!.file)).toEqual({
      version: 'tree-0.1',
      tree: expect.objectContaining({
        id: 'App',
        path: '',
        children: {
          Tree: expect.objectContaining({
            id: 'Tree',
            path: 'Tree',
          }),
          mystack: expect.objectContaining({
            id: 'mystack',
            path: 'mystack',
          }),
        },
      }),
    });
  });
});

function readJson(outdir: string, file: string) {
  return JSON.parse(fs.readFileSync(path.join(outdir, file), 'utf-8'));
}

function isSubtreeReference(x: TreeFile['tree']): x is Extract<TreeFile['tree'], { fileName: string }> {
  return !!(x as any).fileName;
}

function sum(xs: number[]) {
  let ret = 0;
  for (const x of xs) {
    ret += x;
  }
  return ret;
}
