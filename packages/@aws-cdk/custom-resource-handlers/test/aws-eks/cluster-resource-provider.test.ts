
import * as eks from '@aws-sdk/client-eks';
import * as mocks from './cluster-resource-handler-mocks';
import { ClusterResourceHandler } from '../../lib/aws-eks/cluster-resource-handler/cluster';

beforeAll(() => {
  jest.spyOn(console, 'log').mockImplementation();
});

afterAll(() => {
  jest.restoreAllMocks();
});

describe('cluster resource provider', () => {
  beforeEach(() => {
    mocks.reset();
  });

  describe('create', () => {
    test('onCreate: minimal defaults (vpc + role)', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', mocks.MOCK_PROPS));
      await handler.onEvent();

      expect(mocks.actualRequest.configureAssumeRoleRequest).toEqual({
        RoleArn: mocks.MOCK_ASSUME_ROLE_ARN,
        RoleSessionName: 'AWSCDK.EKSCluster.Create.fake-request-id',
      });

      expect(mocks.actualRequest.createClusterRequest).toEqual({
        roleArn: 'arn:of:role',
        resourcesVpcConfig: {
          subnetIds: ['subnet1', 'subnet2'],
          securityGroupIds: ['sg1', 'sg2', 'sg3'],
        },
        name: 'MyResourceId-fakerequestid',
      });
    });

    test('generated cluster name does not exceed 100 characters', async () => {
      // GIVEN
      const req: AWSLambda.CloudFormationCustomResourceCreateEvent = {
        StackId: 'fake-stack-id',
        RequestId: '602c078a-6181-4352-9676-4f00352445aa',
        ResourceType: 'Custom::EKSCluster',
        ServiceToken: 'boom',
        LogicalResourceId: 'hello'.repeat(30), // 150 chars (limit is 100)
        ResponseURL: 'http://response-url',
        RequestType: 'Create',
        ResourceProperties: {
          ServiceToken: 'boom',
          Config: mocks.MOCK_PROPS,
          AssumeRoleArn: mocks.MOCK_ASSUME_ROLE_ARN,
        },
      };

      // WHEN
      const handler = new ClusterResourceHandler(mocks.client, req);
      await handler.onEvent();

      // THEN
      expect(mocks.actualRequest.createClusterRequest?.name?.length).toEqual(100);
      expect(mocks.actualRequest.createClusterRequest?.name).toEqual('hellohellohellohellohellohellohellohellohellohellohellohellohellohe-602c078a6181435296764f00352445aa');
    });

    test('onCreate: explicit cluster name', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', {
        ...mocks.MOCK_PROPS,
        name: 'ExplicitCustomName',
      }));
      await handler.onEvent();

      expect(mocks.actualRequest.createClusterRequest!.name).toEqual('ExplicitCustomName');
    });

    test('with no specific version', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', {
        ...mocks.MOCK_PROPS,
        version: '12.34.56',
      }));
      await handler.onEvent();

      expect(mocks.actualRequest.createClusterRequest!.version).toEqual('12.34.56');
    });

    test('isCreateComplete still not complete if cluster is not ACTIVE', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create'));
      mocks.simulateResponse.describeClusterResponseMockStatus = 'CREATING';
      const resp = await handler.isComplete();
      expect(mocks.actualRequest.describeClusterRequest!.name).toEqual('physical-resource-id');
      expect(resp).toEqual({ IsComplete: false });
    });

    test('isCreateComplete throws if cluster is FAILED', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create'));
      mocks.simulateResponse.describeClusterResponseMockStatus = 'FAILED';
      await expect(handler.isComplete()).rejects.toThrow('Cluster is in a FAILED status');
    });

    test('isUpdateComplete throws if cluster is FAILED', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update'));
      mocks.simulateResponse.describeClusterResponseMockStatus = 'FAILED';
      await expect(handler.isComplete()).rejects.toThrow('Cluster is in a FAILED status');
    });

    test('isCreateComplete is complete when cluster is ACTIVE', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create'));
      mocks.simulateResponse.describeClusterResponseMockStatus = 'ACTIVE';
      const resp = await handler.isComplete();
      expect(resp).toEqual({
        IsComplete: true,
        Data: {
          Name: 'physical-resource-id',
          Endpoint: 'http://endpoint',
          Arn: 'arn:cluster-arn',
          CertificateAuthorityData: 'certificateAuthority-data',
          ClusterSecurityGroupId: '',
          EncryptionConfigKeyArn: '',
          OpenIdConnectIssuerUrl: '',
          OpenIdConnectIssuer: '',
        },
      });
    });

    test('encryption config', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', {
        ...mocks.MOCK_PROPS,
        encryptionConfig: [{ provider: { keyArn: 'aws:kms:key' }, resources: ['secrets'] }],
      }));

      await handler.onEvent();

      expect(mocks.actualRequest.createClusterRequest).toEqual({
        roleArn: 'arn:of:role',
        resourcesVpcConfig: {
          subnetIds: ['subnet1', 'subnet2'],
          securityGroupIds: ['sg1', 'sg2', 'sg3'],
        },
        encryptionConfig: [{ provider: { keyArn: 'aws:kms:key' }, resources: ['secrets'] }],
        name: 'MyResourceId-fakerequestid',
      });
    });

    test('onCreate: deletionProtection is converted from string to boolean', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', {
        ...mocks.MOCK_PROPS,
        deletionProtection: 'true',
      } as any));
      await handler.onEvent();

      expect((mocks.actualRequest.createClusterRequest as any).deletionProtection).toStrictEqual(true);
    });

    test('onCreate: deletionProtection "false" string is converted to boolean false', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', {
        ...mocks.MOCK_PROPS,
        deletionProtection: 'false',
      } as any));
      await handler.onEvent();

      expect((mocks.actualRequest.createClusterRequest as any).deletionProtection).toStrictEqual(false);
    });
  });

  describe('delete', () => {
    test('returns correct physical name', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Delete'));
      const resp = await handler.onEvent();
      expect(mocks.actualRequest.deleteClusterRequest!.name).toEqual('physical-resource-id');
      expect(resp).toEqual({ PhysicalResourceId: 'physical-resource-id' });
    });

    test('onDelete ignores ResourceNotFoundException', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Delete'));
      mocks.simulateResponse.deleteClusterError = new eks.ResourceNotFoundException({
        message: 'Cluster not found',
        $metadata: {
          httpStatusCode: 404,
        },
      });
      await handler.onEvent();
    });

    test('isDeleteComplete returns false as long as describeCluster succeeds', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Delete'));
      const resp = await handler.isComplete();
      expect(mocks.actualRequest.describeClusterRequest!.name).toEqual('physical-resource-id');
      expect(resp.IsComplete).toEqual(false);
    });

    test('isDeleteComplete returns true when describeCluster throws a ResourceNotFound exception', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Delete'));
      mocks.simulateResponse.describeClusterException = new eks.ResourceNotFoundException({
        message: 'Cluster not found',
        $metadata: {
          httpStatusCode: 404,
        },
      });
      const resp = await handler.isComplete();
      expect(resp.IsComplete).toEqual(true);
    });

    test('isDeleteComplete propagates other errors', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Delete'));
      mocks.simulateResponse.describeClusterException = new Error('OtherException');
      let error: any;
      try {
        await handler.isComplete();
      } catch (e) {
        error = e;
      }
      expect(error.message).toEqual('OtherException');
    });
  });

  describe('update', () => {
    test('no change', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', mocks.MOCK_PROPS, mocks.MOCK_PROPS));
      const resp = await handler.onEvent();
      expect(resp).toEqual(undefined);
      expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
      expect(mocks.actualRequest.updateClusterConfigRequest).toEqual(undefined);
      expect(mocks.actualRequest.updateClusterVersionRequest).toEqual(undefined);
    });

    test('isUpdateComplete is not complete when status is not ACTIVE', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update'));
      mocks.simulateResponse.describeClusterResponseMockStatus = 'UPDATING';
      const resp = await handler.isComplete();
      expect(resp.IsComplete).toEqual(false);
    });

    test('isUpdateComplete waits for ACTIVE', async () => {
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update'));
      mocks.simulateResponse.describeClusterResponseMockStatus = 'ACTIVE';
      const resp = await handler.isComplete();
      expect(resp.IsComplete).toEqual(true);
    });

    describe('requires replacement', () => {
      describe('name change', () => {
        test('explicit name change', async () => {
          // GIVEN
          const req = mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            name: 'new-cluster-name-1234',
          }, {
            name: 'old-cluster-name',
          });
          const handler = new ClusterResourceHandler(mocks.client, req);

          // WHEN
          const resp = await handler.onEvent();

          // THEN
          expect(mocks.actualRequest.createClusterRequest!).toEqual({
            name: 'new-cluster-name-1234',
            roleArn: 'arn:of:role',
            resourcesVpcConfig:
            {
              subnetIds: ['subnet1', 'subnet2'],
              securityGroupIds: ['sg1', 'sg2', 'sg3'],
            },
          });
          expect(resp).toEqual({ PhysicalResourceId: 'new-cluster-name-1234' });
        });

        test('from auto-gen name to explicit name', async () => {
          // GIVEN
          const req = mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            name: undefined, // auto-gen
          }, {
            name: 'explicit', // auto-gen
          });

          const handler = new ClusterResourceHandler(mocks.client, req);

          // WHEN
          const resp = await handler.onEvent();

          // THEN
          expect(mocks.actualRequest.createClusterRequest!).toEqual({
            name: 'MyResourceId-fakerequestid',
            roleArn: 'arn:of:role',
            resourcesVpcConfig:
            {
              subnetIds: ['subnet1', 'subnet2'],
              securityGroupIds: ['sg1', 'sg2', 'sg3'],
            },
          });
          expect(resp).toEqual({ PhysicalResourceId: 'MyResourceId-fakerequestid' });
        });
      });

      test('change subnets or security groups order should not trigger an update', async () => {
        const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
          ...mocks.MOCK_PROPS,
          resourcesVpcConfig: {
            subnetIds: ['subnet1', 'subnet2'],
            securityGroupIds: ['sg1', 'sg2'],
          },
        }, {
          ...mocks.MOCK_PROPS,
          resourcesVpcConfig: {
            subnetIds: ['subnet2', 'subnet1'],
            securityGroupIds: ['sg2', 'sg1'],
          },
        }));
        const resp = await handler.onEvent();

        expect(resp).toEqual(undefined);
      });

      test('"roleArn" requires a replacement', async () => {
        const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
          roleArn: 'new-arn',
        }, {
          roleArn: 'old-arn',
        }));
        const resp = await handler.onEvent();

        expect(resp).toEqual({ PhysicalResourceId: 'MyResourceId-fakerequestid' });
        expect(mocks.actualRequest.createClusterRequest).toEqual({
          name: 'MyResourceId-fakerequestid',
          roleArn: 'new-arn',
        });
      });

      test('fails if cluster has an explicit "name" that is the same as the old "name"', async () => {
        // GIVEN
        const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
          roleArn: 'new-arn',
          name: 'explicit-cluster-name',
        }, {
          roleArn: 'old-arn',
          name: 'explicit-cluster-name',
        }));

        // THEN
        let err: any;
        try {
          await handler.onEvent();
        } catch (e) {
          err = e;
        }

        expect(err?.message).toEqual('Cannot replace cluster "explicit-cluster-name" since it has an explicit physical name. Either rename the cluster or remove the "name" configuration');
      });

      test('succeeds if cluster had an explicit "name" and now it does not', async () => {
        // GIVEN
        const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
          roleArn: 'new-arn',
          name: undefined, // auto-gen
        }, {
          roleArn: 'old-arn',
          name: 'explicit-cluster-name',
        }));

        // WHEN
        const resp = await handler.onEvent();

        // THEN
        expect(resp).toEqual({ PhysicalResourceId: 'MyResourceId-fakerequestid' });
        expect(mocks.actualRequest.createClusterRequest).toEqual({
          name: 'MyResourceId-fakerequestid',
          roleArn: 'new-arn',
        });
      });

      test('succeeds if cluster had an explicit "name" and now it has a different name', async () => {
        // GIVEN
        const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
          roleArn: 'new-arn',
          name: 'new-explicit-cluster-name',
        }, {
          roleArn: 'old-arn',
          name: 'old-explicit-cluster-name',
        }));

        // WHEN
        const resp = await handler.onEvent();

        // THEN
        expect(resp).toEqual({ PhysicalResourceId: 'new-explicit-cluster-name' });
        expect(mocks.actualRequest.createClusterRequest).toEqual({
          name: 'new-explicit-cluster-name',
          roleArn: 'new-arn',
        });
      });
    });

    test('encryption config cannot be updated', async () => {
      // GIVEN
      const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
        encryptionConfig: [{ resources: ['secrets'], provider: { keyArn: 'key:arn:1' } }],
      }, {
        encryptionConfig: [{ resources: ['secrets'], provider: { keyArn: 'key:arn:2' } }],
      }));

      // WHEN
      let error: any;
      try {
        await handler.onEvent();
      } catch (e) {
        error = e;
      }

      // THEN
      expect(error).toBeDefined();
      expect(error.message).toEqual('Cannot update cluster encryption configuration');
    });

    describe('isUpdateComplete with EKS update ID', () => {
      test('with "Failed" status', async () => {
        const event = mocks.newRequest('Update');
        const isCompleteHandler = new ClusterResourceHandler(mocks.client, {
          ...event,
          EksUpdateId: 'foobar',
        });

        mocks.simulateResponse.describeUpdateResponseMockStatus = 'Failed';
        mocks.simulateResponse.describeUpdateResponseMockErrors = [
          {
            errorMessage: 'errorMessageMock',
            errorCode: eks.ErrorCode.UNKNOWN,
            resourceIds: [
              'foo', 'bar',
            ],
          },
        ];

        let error: any;
        try {
          await isCompleteHandler.isComplete();
        } catch (e) {
          error = e;
        }
        expect(error).toBeDefined();
        expect(mocks.actualRequest.describeUpdateRequest).toEqual({ name: 'physical-resource-id', updateId: 'foobar' });
        expect(error.message).toEqual('cluster update id "foobar" failed with errors: [{"errorMessage":"errorMessageMock","errorCode":"Unknown","resourceIds":["foo","bar"]}]');
      });

      test('with "InProgress" status, returns IsComplete=false', async () => {
        const event = mocks.newRequest('Update');
        const isCompleteHandler = new ClusterResourceHandler(mocks.client, {
          ...event,
          EksUpdateId: 'foobar',
        });

        mocks.simulateResponse.describeUpdateResponseMockStatus = 'InProgress';

        const response = await isCompleteHandler.isComplete();

        expect(mocks.actualRequest.describeUpdateRequest).toEqual({ name: 'physical-resource-id', updateId: 'foobar' });
        expect(response.IsComplete).toEqual(false);
      });

      test('with "Successful" status, returns IsComplete=true with "Data"', async () => {
        const event = mocks.newRequest('Update');
        const isCompleteHandler = new ClusterResourceHandler(mocks.client, {
          ...event,
          EksUpdateId: 'foobar',
        });

        mocks.simulateResponse.describeUpdateResponseMockStatus = 'Successful';

        const response = await isCompleteHandler.isComplete();

        expect(mocks.actualRequest.describeUpdateRequest).toEqual({ name: 'physical-resource-id', updateId: 'foobar' });
        expect(response).toEqual({
          IsComplete: true,
          Data: {
            Name: 'physical-resource-id',
            Endpoint: 'http://endpoint',
            Arn: 'arn:cluster-arn',
            CertificateAuthorityData: 'certificateAuthority-data',
            ClusterSecurityGroupId: '',
            EncryptionConfigKeyArn: '',
            OpenIdConnectIssuerUrl: '',
            OpenIdConnectIssuer: '',
          },
        });
      });
    });

    describe('in-place', () => {
      describe('version change', () => {
        test('from undefined to a specific value', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '12.34',
          }, {
            version: undefined,
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterVersionRequest!).toEqual({
            name: 'physical-resource-id',
            version: '12.34',
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('from a specific value to another value', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '2.0',
          }, {
            version: '1.1',
          }));

          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterVersionRequest!).toEqual({
            name: 'physical-resource-id',
            version: '2.0',
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('to a new value that is already the current version', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '1.0',
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual(undefined);
          expect(mocks.actualRequest.describeClusterRequest).toEqual({ name: 'physical-resource-id' });
          expect(mocks.actualRequest.updateClusterVersionRequest).toEqual(undefined);
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('fails from specific value to undefined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: undefined,
          }, {
            version: '1.2',
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }

          expect(error.message).toEqual('Cannot remove cluster version configuration. Current version is 1.2');
        });

        test('subnets or security groups does not require a replacement', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            resourcesVpcConfig: {
              subnetIds: ['subnet1', 'subnet2'],
              securityGroupIds: ['sg1'],
            },
          }, {
            ...mocks.MOCK_PROPS,
            resourcesVpcConfig: {
              subnetIds: ['subnet1'],
              securityGroupIds: ['sg2'],
            },
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: 'MockEksUpdateStatusId' });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('update both version and authModeUpdate with accessConfig undefined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '2.0',
            accessConfig: { authenticationMode: 'API' },
          }, {
            version: '1.0',
            accessConfig: undefined,
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });

        test('both version and authModeUpdate defined and modify both of them', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '2.0',
            accessConfig: { authenticationMode: 'API' },
          }, {
            version: '1.0',
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });

        test('update both version and logging with logging undefined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '2.0',
            logging: {
              clusterLogging: [
                {
                  types: ['api'],
                  enabled: true,
                },
              ],
            },
          }, {
            version: '1.0',
            logging: undefined,
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });

        test('both version and logging defined and modify both of them', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            version: '2.0',
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
          }, {
            version: '1.0',
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager'],
                  enabled: true,
                },
              ],
            },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
      });
      describe('assessConfig change', () => {
        test('from undefined to a specific value', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          }, {
            accessConfig: { authenticationMode: undefined },
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('from a specific value to another value', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: { authenticationMode: 'API' },
          }, {
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          }));

          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            accessConfig: { authenticationMode: 'API' },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('fails from any defined value to undefined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: undefined,
          }, {
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }

          expect(error.message).toEqual('Cannot fallback authenticationMode from defined to undefined');
        });
        test('fails from API to undefined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: { authenticationMode: undefined },
          }, {
            accessConfig: { authenticationMode: 'API' },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }

          expect(error.message).toEqual('Cannot fallback authenticationMode from API to undefined');
        });
        test('fails from API to API_AND_CONFIG_MAP', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          }, {
            accessConfig: { authenticationMode: 'API' },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }

          expect(error.message).toEqual('Cannot fallback authenticationMode from API to API_AND_CONFIG_MAP');
        });
        test('fails from undefined to API', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: { authenticationMode: 'API' },
          }, {
            accessConfig: { authenticationMode: undefined },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }

          expect(error.message).toEqual('Cannot update from undefined(CONFIG_MAP) to API');
        });

        test('fails from CONFIG_MAP to API', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            accessConfig: { authenticationMode: 'API' },
          }, {
            accessConfig: { authenticationMode: 'CONFIG_MAP' },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }

          expect(error.message).toEqual('Cannot update from CONFIG_MAP to API');
        });
        test('from undefined to both logging and authenticationMode enabled', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            accessConfig: { authenticationMode: 'API' },
          }, {
            logging: undefined,
            accessConfig: undefined,
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
        test('from undefined to both access and authenticationMode enabled', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
            accessConfig: { authenticationMode: 'API' },
          }, {
            resourcesVpcConfig: undefined,
            accessConfig: undefined,
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
        test('update both EndpointAccessUpdate and AuthModeUpdate with accessConfig undefined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            resourcesVpcConfig: {
              endpointPrivateAccess: false,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
            accessConfig: { authenticationMode: 'API' },
          }, {
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
            accessConfig: undefined,
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
        test('update both EndpointAccessUpdate and AuthModeUpdate with accessConfig defined', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            resourcesVpcConfig: {
              endpointPrivateAccess: false,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
            accessConfig: { authenticationMode: 'API' },
          }, {
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
            accessConfig: { authenticationMode: 'API_AND_CONFIG_MAP' },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
      });

      describe('logging or access change', () => {
        test('from undefined to partial logging enabled', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            logging: {
              clusterLogging: [
                {
                  types: ['api'],
                  enabled: true,
                },
              ],
            },
          }, {
            logging: undefined,
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            logging: {
              clusterLogging: [
                {
                  types: ['api'],
                  enabled: true,
                },
              ],
            },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('from partial vpc configuration to only private access enabled', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            resourcesVpcConfig: {
              securityGroupIds: ['sg1', 'sg2', 'sg3'],
              endpointPrivateAccess: true,
            },
          }, {
            resourcesVpcConfig: {
              securityGroupIds: ['sg1', 'sg2', 'sg3'],
            },
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            logging: undefined,
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: undefined,
              publicAccessCidrs: undefined,
            },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('from undefined to both logging and access fully enabled', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
          }, {
            logging: undefined,
            resourcesVpcConfig: undefined,
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
        test('both logging and access defined and modify both of them', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: false,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
          }, {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
          }));
          let error: any;
          try {
            await handler.onEvent();
          } catch (e) {
            error = e;
          }
          expect(error.message).toEqual('Only one type of update - replaceName, updateVpc, updateAccess, replaceRole, updateVersion, updateEncryption, updateLogging, updateAuthMode, updateBootstrapClusterCreatorAdminPermissions, updateBootstrapSelfManagedAddons, updateDeletionProtection, updateControlPlaneScalingConfig can be allowed');
        });
        test('Given logging enabled and unchanged, updating the only publicAccessCidrs is allowed ', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['1.2.3.4/32'],
            },
          }, {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['0.0.0.0/0'],
            },
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: 'MockEksUpdateStatusId' });
        });
        test('Given logging enabled and unchanged, updating publicAccessCidrs from one to multiple entries is allowed ', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['2.4.6.0/24', '1.2.3.4/32', '3.3.3.3/32'],
            },
          }, {
            logging: {
              clusterLogging: [
                {
                  types: ['api', 'audit', 'authenticator', 'controllerManager', 'scheduler'],
                  enabled: true,
                },
              ],
            },
            resourcesVpcConfig: {
              endpointPrivateAccess: true,
              endpointPublicAccess: true,
              publicAccessCidrs: ['2.4.6.0/24'],
            },
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: 'MockEksUpdateStatusId' });
        });
      });
      describe('tag updates', () => {
        test('updates are in-place', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            // this is the new props
            ...mocks.MOCK_PROPS,
            tags: {
              foo: 'bar',
              hello: 'world',
            },
          }, {
            // this is the old props
            ...mocks.MOCK_PROPS,
            tags: {
              foo: 'bar',
            },
          }));
          await handler.onEvent();
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
          expect(mocks.actualRequest.tagResourceRequest).toEqual({
            resourceArn: 'arn:cluster-arn',
            tags: {
              hello: 'world',
            },
          });
          expect(mocks.actualRequest.untagResourceRequest).toEqual(undefined);
        });
        test('update a tag along with a removal', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            // this is the new props
            ...mocks.MOCK_PROPS,
            tags: {
              foo: 'world',
            },
          }, {
            // this is the old props
            ...mocks.MOCK_PROPS,
            tags: {
              foo: 'bar',
              hello: 'world',
            },
          }));
          await handler.onEvent();
          expect(mocks.actualRequest.tagResourceRequest).toEqual({
            resourceArn: 'arn:cluster-arn',
            tags: {
              foo: 'world',
            },
          });
          expect(mocks.actualRequest.untagResourceRequest).toEqual({
            resourceArn: 'arn:cluster-arn',
            tagKeys: ['hello'],
          });
        });
        test('remove all tags', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            // this is the new props
            ...mocks.MOCK_PROPS,
          }, {
            // this is the old props
            ...mocks.MOCK_PROPS,
            tags: {
              foo: 'bar',
              hello: 'world',
            },
          }));
          await handler.onEvent();
          expect(mocks.actualRequest.tagResourceRequest).toEqual(undefined);
          expect(mocks.actualRequest.untagResourceRequest).toEqual({
            resourceArn: 'arn:cluster-arn',
            tagKeys: ['foo', 'hello'],
          });
        });
        test('add tags after creation of cluster', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            // this is the new props
            ...mocks.MOCK_PROPS,
            tags: {
              foo: 'bar',
              hello: 'world',
            },
          }, {
            // this is the old props
            ...mocks.MOCK_PROPS,
          }));
          await handler.onEvent();
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
          expect(mocks.actualRequest.tagResourceRequest).toEqual({
            resourceArn: 'arn:cluster-arn',
            tags: {
              foo: 'bar',
              hello: 'world',
            },
          });
          expect(mocks.actualRequest.untagResourceRequest).toEqual(undefined);
        });
        test('updates to bootstrapSelfManagedAddons requires a replacement', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            bootstrapSelfManagedAddons: false,
          }, {
            ...mocks.MOCK_PROPS,
            bootstrapSelfManagedAddons: true,
          }));
          await handler.onEvent();
          expect(mocks.actualRequest.createClusterRequest!).toEqual({
            bootstrapSelfManagedAddons: false,
            name: 'MyResourceId-fakerequestid',
            roleArn: 'arn:of:role',
            resourcesVpcConfig:
            {
              subnetIds: ['subnet1', 'subnet2'],
              securityGroupIds: ['sg1', 'sg2', 'sg3'],
            },
          });
        });
      });
      describe('deletionProtection change', () => {
        test('from undefined to true triggers updateClusterConfig', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            deletionProtection: true,
          } as any, {
            ...mocks.MOCK_PROPS,
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect((mocks.actualRequest.updateClusterConfigRequest as any).deletionProtection).toEqual(true);
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('from true to false triggers updateClusterConfig', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            deletionProtection: false,
          } as any, {
            ...mocks.MOCK_PROPS,
            deletionProtection: true,
          } as any));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect((mocks.actualRequest.updateClusterConfigRequest as any).deletionProtection).toEqual(false);
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('no change does not trigger update', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            deletionProtection: true,
          } as any, {
            ...mocks.MOCK_PROPS,
            deletionProtection: true,
          } as any));
          const resp = await handler.onEvent();
          expect(resp).toEqual(undefined);
          expect(mocks.actualRequest.updateClusterConfigRequest).toEqual(undefined);
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });
      });
      describe('controlPlaneScalingConfig change', () => {
        test('from undefined to a provisioned tier triggers updateClusterConfig', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          } as any, {
            ...mocks.MOCK_PROPS,
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('from one provisioned tier to another triggers updateClusterConfig', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-4xl' },
          } as any, {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          } as any));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            controlPlaneScalingConfig: { tier: 'tier-4xl' },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('removing the configuration falls back to the standard tier', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
          }, {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          } as any));
          const resp = await handler.onEvent();
          expect(resp).toEqual({ EksUpdateId: mocks.MOCK_UPDATE_STATUS_ID });
          expect(mocks.actualRequest.updateClusterConfigRequest!).toEqual({
            name: 'physical-resource-id',
            controlPlaneScalingConfig: { tier: 'standard' },
          });
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('no change does not trigger update', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          } as any, {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          } as any));
          const resp = await handler.onEvent();
          expect(resp).toEqual(undefined);
          expect(mocks.actualRequest.updateClusterConfigRequest).toEqual(undefined);
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('setting the standard tier explicitly does not trigger update', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Update', {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'standard' },
          } as any, {
            ...mocks.MOCK_PROPS,
          }));
          const resp = await handler.onEvent();
          expect(resp).toEqual(undefined);
          expect(mocks.actualRequest.updateClusterConfigRequest).toEqual(undefined);
          expect(mocks.actualRequest.createClusterRequest).toEqual(undefined);
        });

        test('is passed through on create', async () => {
          const handler = new ClusterResourceHandler(mocks.client, mocks.newRequest('Create', {
            ...mocks.MOCK_PROPS,
            controlPlaneScalingConfig: { tier: 'tier-xl' },
          } as any));
          await handler.onEvent();
          expect((mocks.actualRequest.createClusterRequest as any).controlPlaneScalingConfig).toEqual({ tier: 'tier-xl' });
        });
      });
    });
  });
});
