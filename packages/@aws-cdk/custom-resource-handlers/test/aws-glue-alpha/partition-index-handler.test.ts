import { CreatePartitionIndexCommand, DeletePartitionIndexCommand, GetPartitionIndexesCommand, GlueClient } from '@aws-sdk/client-glue';
import { mockClient } from 'aws-sdk-client-mock';
import { isComplete, onEvent } from '../../lib/aws-glue-alpha/partition-index-handler';

const glueMock = mockClient(GlueClient);

let oldConsoleLog: any;

beforeAll(() => {
  // eslint-disable-next-line @typescript-eslint/unbound-method
  oldConsoleLog = global.console.log;
  global.console.log = jest.fn();
});

afterAll(() => {
  global.console.log = oldConsoleLog;
});

afterEach(() => {
  glueMock.reset();
});

const baseProperties = {
  DatabaseName: 'my-database',
  TableName: 'my-table',
  IndexName: 'my-index',
  Keys: ['year', 'month'],
};

function event(requestType: string, extra: Record<string, any> = {}) {
  return {
    RequestType: requestType,
    ResourceProperties: baseProperties,
    ...extra,
  };
}

/** Build a mocked class error that matches the AWS SDK's `error.name` discrimination. */
function awsError(name: string): Error {
  const err = new Error(name);
  err.name = name;
  return err;
}

describe('onEvent', () => {
  describe('Create', () => {
    test('creates the partition index and returns the index name as the physical id', async () => {
      glueMock.on(CreatePartitionIndexCommand).resolves({});

      const result = await onEvent(event('Create'));

      expect(result).toEqual({ PhysicalResourceId: 'my-index' });
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(1);
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)[0].args[0].input).toEqual({
        DatabaseName: 'my-database',
        TableName: 'my-table',
        PartitionIndex: { IndexName: 'my-index', Keys: ['year', 'month'] },
      });
    });

    test('swallows AlreadyExistsException when the existing index has matching keys', async () => {
      glueMock.on(CreatePartitionIndexCommand).rejects(awsError('AlreadyExistsException'));
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{
          IndexName: 'my-index',
          Keys: [{ Name: 'year', Type: 'string' }, { Name: 'month', Type: 'string' }],
          IndexStatus: 'ACTIVE',
        }],
      });

      const result = await onEvent(event('Create'));

      expect(result).toEqual({ PhysicalResourceId: 'my-index' });
    });

    test('throws when the existing index has different keys', async () => {
      glueMock.on(CreatePartitionIndexCommand).rejects(awsError('AlreadyExistsException'));
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{
          IndexName: 'my-index',
          Keys: [{ Name: 'year', Type: 'string' }, { Name: 'day', Type: 'string' }],
          IndexStatus: 'ACTIVE',
        }],
      });

      await expect(onEvent(event('Create'))).rejects.toThrow(
        /Partition index my-index already exists on my-database/,
      );
    });

    test('rethrows any other error', async () => {
      glueMock.on(CreatePartitionIndexCommand).rejects(awsError(' OperationTimeoutException '));

      await expect(onEvent(event('Create'))).rejects.toThrow(' OperationTimeoutException ');
    });
  });

  describe('Update', () => {
    test('no-ops when the keys are unchanged', async () => {
      const result = await onEvent(event('Update', {
        OldResourceProperties: { ...baseProperties },
      }));

      expect(result).toEqual({ PhysicalResourceId: 'my-index' });
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(0);
    });

    test('kicks off deletion of the old index when the keys change', async () => {
      glueMock.on(DeletePartitionIndexCommand).resolves({});

      const result = await onEvent(event('Update', {
        OldResourceProperties: { ...baseProperties, Keys: ['year'] },
      }));

      expect(result).toEqual({ PhysicalResourceId: 'my-index' });
      expect(glueMock.commandCalls(DeletePartitionIndexCommand)).toHaveLength(1);
      expect(glueMock.commandCalls(DeletePartitionIndexCommand)[0].args[0].input).toEqual({
        DatabaseName: 'my-database',
        TableName: 'my-table',
        IndexName: 'my-index',
      });
      // No CreatePartitionIndex here: recreation is driven by isComplete once the
      // old index has finished deleting.
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(0);
    });

    test('swallows EntityNotFoundException while kicking off deletion on a key change', async () => {
      glueMock.on(DeletePartitionIndexCommand).rejects(awsError('EntityNotFoundException'));

      await expect(onEvent(event('Update', {
        OldResourceProperties: { ...baseProperties, Keys: ['year'] },
      }))).resolves.toEqual({ PhysicalResourceId: 'my-index' });
    });
  });

  describe('Delete', () => {
    test('deletes the partition index', async () => {
      glueMock.on(DeletePartitionIndexCommand).resolves({});

      await onEvent(event('Delete'));

      expect(glueMock.commandCalls(DeletePartitionIndexCommand)).toHaveLength(1);
      expect(glueMock.commandCalls(DeletePartitionIndexCommand)[0].args[0].input).toEqual({
        DatabaseName: 'my-database',
        TableName: 'my-table',
        IndexName: 'my-index',
      });
    });

    test('swallows EntityNotFoundException', async () => {
      glueMock.on(DeletePartitionIndexCommand).rejects(awsError('EntityNotFoundException'));

      await expect(onEvent(event('Delete'))).resolves.toEqual({});
    });

    test('rethrows any other error', async () => {
      glueMock.on(DeletePartitionIndexCommand).rejects(awsError('InternalServiceException'));

      await expect(onEvent(event('Delete'))).rejects.toThrow('InternalServiceException');
    });
  });
});

describe('isComplete', () => {
  describe('Create', () => {
    test('is not complete while the index is missing', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({ PartitionIndexDescriptorList: [] });

      await expect(isComplete(event('Create'))).resolves.toEqual({ IsComplete: false });
    });

    test('is not complete while the index is CREATING', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{ IndexName: 'my-index', Keys: [], IndexStatus: 'CREATING' }],
      });

      await expect(isComplete(event('Create'))).resolves.toEqual({ IsComplete: false });
    });

    test('is complete once the index is ACTIVE', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{ IndexName: 'my-index', Keys: [], IndexStatus: 'ACTIVE' }],
      });

      await expect(isComplete(event('Create'))).resolves.toEqual({ IsComplete: true });
    });

    test('matches the index case-insensitively (Glue lowercases names)', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{ IndexName: 'my-index', Keys: [], IndexStatus: 'ACTIVE' }],
      });

      await expect(isComplete(event('Create', {
        ResourceProperties: { ...baseProperties, IndexName: 'MY-INDEX' },
      }))).resolves.toEqual({ IsComplete: true });
    });

    test('throws with backfill errors when the index is in an unexpected state', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{
          IndexName: 'my-index',
          Keys: [],
          IndexStatus: 'FAILED',
          BackfillErrors: [{ Code: 'INTERNAL_ERROR' }],
        }],
      });

      await expect(isComplete(event('Create'))).rejects.toThrow(
        /Partition index my-index in unexpected state: FAILED\. Backfill errors:/,
      );
    });
  });

  describe('Delete', () => {
    test('is complete when the index no longer exists', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({ PartitionIndexDescriptorList: [] });

      await expect(isComplete(event('Delete'))).resolves.toEqual({ IsComplete: true });
    });

    test('is not complete while the index is still DELETING', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{ IndexName: 'my-index', Keys: [], IndexStatus: 'DELETING' }],
      });

      await expect(isComplete(event('Delete'))).resolves.toEqual({ IsComplete: false });
    });
  });

  describe('Update with changed keys', () => {
    const changedEvent = () => event('Update', {
      OldResourceProperties: { ...baseProperties, Keys: ['year'] },
    });

    test('waits while the old index is still present with its old keys', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{
          IndexName: 'my-index',
          Keys: [{ Name: 'year', Type: 'string' }],
          IndexStatus: 'ACTIVE',
        }],
      });

      await expect(isComplete(changedEvent())).resolves.toEqual({ IsComplete: false });
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(0);
    });

    test('creates the replacement once the old index is gone', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({ PartitionIndexDescriptorList: [] });
      glueMock.on(CreatePartitionIndexCommand).resolves({});

      await expect(isComplete(changedEvent())).resolves.toEqual({ IsComplete: false });
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(1);
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)[0].args[0].input).toEqual({
        DatabaseName: 'my-database',
        TableName: 'my-table',
        PartitionIndex: { IndexName: 'my-index', Keys: ['year', 'month'] },
      });
    });

    test('does not recreate while the replacement is CREATING with the new keys', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{
          IndexName: 'my-index',
          Keys: [{ Name: 'year', Type: 'string' }, { Name: 'month', Type: 'string' }],
          IndexStatus: 'CREATING',
        }],
      });

      await expect(isComplete(changedEvent())).resolves.toEqual({ IsComplete: false });
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(0);
    });

    test('is complete once the replacement carries the new keys and is ACTIVE', async () => {
      glueMock.on(GetPartitionIndexesCommand).resolves({
        PartitionIndexDescriptorList: [{
          IndexName: 'my-index',
          Keys: [{ Name: 'year', Type: 'string' }, { Name: 'month', Type: 'string' }],
          IndexStatus: 'ACTIVE',
        }],
      });

      await expect(isComplete(changedEvent())).resolves.toEqual({ IsComplete: true });
      expect(glueMock.commandCalls(CreatePartitionIndexCommand)).toHaveLength(0);
    });
  });
});
