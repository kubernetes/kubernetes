/*
 * Custom resource handler for Glue partition indexes.
 *
 * Glue has no UpdatePartitionIndex API, so this handler emulates create / update /
 * delete on top of Create/Delete + polling. It is split across the two functions the
 * provider framework drives: `onEvent` runs ONCE per CloudFormation request; `isComplete`
 * is POLLED (~every 10s, up to totalTimeout) until it returns IsComplete=true or throws.
 *
 * onEvent (one-shot):
 *   Create           -> CreatePartitionIndex (AlreadyExists w/ matching keys = ok)
 *   Update, no-op    -> nothing (neither table nor keys changed)
 *   Update, table    -> nothing here; isComplete creates on the new table
 *   Update, keys     -> DeletePartitionIndex (old); isComplete recreates with new keys
 *   Delete           -> DeletePartitionIndex
 * NOTE: onEvent never CREATES on Update - all (re)creation is deferred to isComplete.
 *
 * isComplete (polled loop). Each poll looks up the index, then returns the first
 * matching case below. "changed" = table or keys changed on an Update; DONE returns
 * IsComplete=true, "wait" returns IsComplete=false (poll again), otherwise it throws.
 *
 *   1. Delete request      -> DONE if the index is gone, else wait.
 *   2. changed + no index  -> CreatePartitionIndex (swallow AlreadyExists), then wait.
 *   3. changed + old keys  -> wait (old index still ACTIVE or mid-delete).
 *   4. changed + new keys  -> fall through to the status check (case 5).
 *   5. index status:  ACTIVE -> DONE ;  CREATING/DELETING/missing -> wait ;
 *                     FAILED/other -> throw (deploy fails).
 */
// @aws-sdk/* modules available at runtime for lambdas >= Node18
// eslint-disable-next-line import/no-extraneous-dependencies
import { GlueClient, CreatePartitionIndexCommand, DeletePartitionIndexCommand, GetPartitionIndexesCommand } from '@aws-sdk/client-glue';

class PartitionIndexError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PartitionIndexError';
  }
}

const glue = new GlueClient({});

async function findPartitionIndex(DatabaseName: string, TableName: string, IndexName: string) {
  // Glue caps partition indexes at 3/table, so results always fit in one page
  const resp = await glue.send(new GetPartitionIndexesCommand({ DatabaseName, TableName }));
  // Glue lowercases index names, so compare case-insensitively
  return (resp.PartitionIndexDescriptorList || []).find(
    (i) => i.IndexName?.toLowerCase() === IndexName.toLowerCase(),
  );
}

function createPartitionIndex(DatabaseName: string, TableName: string, IndexName: string, Keys: string[]) {
  return glue.send(new CreatePartitionIndexCommand({
    DatabaseName,
    TableName,
    PartitionIndex: { IndexName, Keys },
  }));
}

export async function onEvent(event: any) {
  const { DatabaseName, TableName, IndexName, Keys } = event.ResourceProperties;

  if (event.RequestType === 'Create') {
    try {
      await createPartitionIndex(DatabaseName, TableName, IndexName, Keys);
    } catch (e: any) {
      // The index may already exist if it was created out-of-band or by a previous
      // resource being replaced (CloudFormation creates the replacement before deleting
      // the old resource). Treat this as success and let isComplete verify its state.
      if (e.name === 'AlreadyExistsException' || e.name === 'ResourceNumberLimitExceededException') {
        const existing = await findPartitionIndex(DatabaseName, TableName, IndexName);
        const existingKeys = existing?.Keys?.map((k: any) => k.Name);
        if (!existing || JSON.stringify(existingKeys) !== JSON.stringify(Keys)) {
          throw new PartitionIndexError(
            `Partition index ${IndexName} already exists on ${DatabaseName}.${TableName} with different or reordered keys ` +
              `(existing: ${JSON.stringify(existingKeys)}, requested: ${JSON.stringify(Keys)}). Delete the existing index first.`,
          );
        }
        console.log(`Partition index ${IndexName} already exists with matching keys - reusing`);
      } else {
        throw e;
      }
    }
    return { PhysicalResourceId: IndexName };
  } else if (event.RequestType === 'Update') {
    const oldProps = event.OldResourceProperties ?? {};
    const tableChanged =
      oldProps.DatabaseName !== DatabaseName || oldProps.TableName !== TableName;
    const keysChanged = JSON.stringify(oldProps.Keys) !== JSON.stringify(Keys);

    if (!tableChanged && !keysChanged) {
      // No actionable change.
      return { PhysicalResourceId: IndexName };
    }

    // If the target table changed, it means the old table is being replaced (a Glue
    // table's name cannot change in place), so the index must be created on the NEW
    // table, which has no index yet - isComplete does that. There is nothing to delete
    // here: a partition index is metadata on the table, so Glue deletes the old index
    // together with the old table (and the handler role is only scoped to the new
    // table, so it cannot touch the old one anyway).
    if (tableChanged) {
      return { PhysicalResourceId: IndexName };
    }

    // Keys changed on the same table. The index name is stable (e.g. tokenized keys,
    // or a user-supplied name), and Glue has no update API and will not allow two
    // indexes with the same name, so the old index must be deleted and recreated with
    // the new keys. Deletion is asynchronous, so we only kick it off here; isComplete
    // drives the delete -> recreate -> ACTIVE sequence.
    //
    // This is intentionally NOT atomic: there is a window where the index does not
    // exist. Create-before-delete is impossible here - the stable name would collide,
    // and a table already at the 3-index cap could not hold a transient 4th. The
    // window is low-risk: a partition index is query-optimization metadata, not data,
    // so the only impact is degraded query performance until the new index is ACTIVE.
    // If the recreate fails, CloudFormation rollback re-runs this same path with the
    // keys swapped and restores the original index.
    try {
      await glue.send(new DeletePartitionIndexCommand({ DatabaseName, TableName, IndexName }));
    } catch (e: any) {
      if (e.name !== 'EntityNotFoundException') {
        throw e;
      }
    }
    return { PhysicalResourceId: IndexName };
  } else if (event.RequestType === 'Delete') {
    try {
      await glue.send(new DeletePartitionIndexCommand({
        DatabaseName,
        TableName,
        IndexName,
      }));
    } catch (e: any) {
      if (e.name === 'EntityNotFoundException') {
        console.log(`Partition index ${IndexName} not found on ${DatabaseName}.${TableName} - may have been deleted out-of-band`);
      } else {
        throw e;
      }
    }
  }

  return {};
}

export async function isComplete(event: any) {
  const { DatabaseName, TableName, IndexName, Keys } = event.ResourceProperties;
  const index = await findPartitionIndex(DatabaseName, TableName, IndexName);

  if (event.RequestType === 'Delete') {
    return { IsComplete: !index };
  }

  // On an Update, onEvent may have kicked off deletion of the old (same-named)
  // index on this table (keys changed), or the target table itself changed and the
  // index does not exist on the new table yet. Either way the index has to be
  // (re)created here: wait for any old same-named index to disappear, create the
  // replacement, then fall through to the status checks below once it carries the
  // requested keys.
  const oldProps = event.OldResourceProperties ?? {};
  const tableChanged =
    event.RequestType === 'Update' &&
    (oldProps.DatabaseName !== DatabaseName || oldProps.TableName !== TableName);
  const keysChanged =
    event.RequestType === 'Update' &&
    JSON.stringify(oldProps.Keys) !== JSON.stringify(Keys);

  if (tableChanged || keysChanged) {
    const existingKeys = index?.Keys?.map((k: any) => k.Name);
    const isNewIndex = index !== undefined && JSON.stringify(existingKeys) === JSON.stringify(Keys);

    if (!index) {
      // The index does not (yet) exist on the target table -> create it. This is a
      // fresh table (table changed) or the old same-named index has finished
      // deleting (keys changed). GetPartitionIndexes is eventually consistent, so a
      // create issued on a previous poll may not be visible yet; swallow
      // AlreadyExistsException so a repeat poll doesn't fail the deployment. The next
      // poll that sees the index (CREATING) falls through to the status checks below.
      try {
        await createPartitionIndex(DatabaseName, TableName, IndexName, Keys);
      } catch (e: any) {
        if (e.name !== 'AlreadyExistsException') {
          // keys are validated by the isNewIndex guard on the next poll
          throw e;
        }
      }
      return { IsComplete: false };
    }
    if (!isNewIndex) {
      // Still the OLD index (ACTIVE with old keys, or DELETING) -> keep waiting.
      return { IsComplete: false };
    }
    // The index now carries the requested keys -> evaluate its status below.
  }

  if (!index) return { IsComplete: false };
  if (index.IndexStatus === 'ACTIVE') return { IsComplete: true };
  if (index.IndexStatus === 'CREATING') return { IsComplete: false };
  if (index.IndexStatus === 'DELETING') return { IsComplete: false };

  const errorDetails = index.BackfillErrors?.length
    ? ` Backfill errors: ${JSON.stringify(index.BackfillErrors)}`
    : '';
  throw new PartitionIndexError(`Partition index ${IndexName} in unexpected state: ${index.IndexStatus}.${errorDetails}`);
}
