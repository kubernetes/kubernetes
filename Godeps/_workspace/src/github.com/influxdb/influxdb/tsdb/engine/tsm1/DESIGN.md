# File Structure

A TSM file is composed for four sections: header, blocks, index and the footer.

```
┌────────┬────────────────────────────────────┬─────────────┬──────────────┐
│ Header │               Blocks               │    Index    │    Footer    │
│5 bytes │              N bytes               │   N bytes   │   4 bytes    │
└────────┴────────────────────────────────────┴─────────────┴──────────────┘
```
Header is composed of a magic number to identify the file type and a version number.

```
┌───────────────────┐
│      Header       │
├─────────┬─────────┤
│  Magic  │ Version │
│ 4 bytes │ 1 byte  │
└─────────┴─────────┘
```

Blocks are sequences of block CRC32 and data.  The block data is opaque to the file.  The CRC32 is used for recovery to ensure blocks have not been corrupted due to bugs outside of our control.  The length of the blocks is stored in the index.

```
┌───────────────────────────────────────────────────────────┐
│                          Blocks                           │
├───────────────────┬───────────────────┬───────────────────┤
│      Block 1      │      Block 2      │      Block N      │
├─────────┬─────────┼─────────┬─────────┼─────────┬─────────┤
│  CRC    │  Data   │  CRC    │  Data   │  CRC    │  Data   │
│ 4 bytes │ N bytes │ 4 bytes │ N bytes │ 4 bytes │ N bytes │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

Following the blocks is the index for the blocks in the file.  The index is composed of a sequence of index entries ordered lexicographically by key and then by time.  Each index entry starts with a key length and key followed by a count of the number of blocks in the file.  Each block entry is composed of the min and max time for the block, the offset into the file where the block is located and the size of the block.

The index structure can provide efficient access to all blocks as well as the ability to determine the cost associated with acessing given key.  Given a key and timestamp, we know exactly which file contains the block for that timestamp as well as where that block resides and how much data to read to retrieve the block.  If we know we need to read all or multiple blocks in a file, we can use the size to determine how much to read in a given IO.

_TBD: The block length stored in the block data could probably be dropped since we store it in the index._

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                   Index                                    │
├─────────┬─────────┬──────┬───────┬─────────┬─────────┬────────┬────────┬───┤
│ Key Len │   Key   │ Type │ Count │Min Time │Max Time │ Offset │  Size  │...│
│ 2 bytes │ N bytes │1 byte│2 bytes│ 8 bytes │ 8 bytes │8 bytes │4 bytes │   │
└─────────┴─────────┴──────┴───────┴─────────┴─────────┴────────┴────────┴───┘
```

The last section is the footer that stores the offset of the start of the index.

```
┌─────────┐
│ Footer  │
├─────────┤
│Index Ofs│
│ 8 bytes │
└─────────┘
```

# File System Layout

The file system is organized a directory per shard where each shard is integer number.  Within the shard dir exists a set of other directories and files:

* wal Dir - Contains a set numerically increasing files WAL segment files name ######.wal.  The wal dir will be a separate location from the TSM data files so that different types can be used if necessary.
* TSM files - A set of numerically increasing TSM files containing compressed series data.
* Tombstone files - Files named after the corresponding TSM file as #####.tombstone.  These contain measurement and series keys that have been deleted.  These are removed during compactions.

# Data Flow

Writes are appended to the current WAL segment and are also added to the Cache.  Each WAL segment is size bounded and rolls-over to a new file after it fills up.  The cache is also size bounded and older entries are evicted as new entries are added to maintain the size.  The WAL and Cache are separate entities and do not interact with each other.  The Engine coordinates the writes to both.

When WAL segments fill up and closed, the Compactor reads the WAL entries and combines then with one or more existing TSM files.  This process runs continously until all WAL files are compacted and there is a minimum number of TSM files.  As each TSM file is completed, it is loaded and referenced by the FileStore.

Queries are executed by constructing Cursors for keys.  The Cursors iterate of slices of Values.  When the current Values are exhausted, a cursor requests a the next set of Values from the Engine.  The Engine returns a slice of Values by querying the FileStore and Cache.  The Values in the Cache are overlayed on top of the values returned from the FileStore.  The FileStore reads and decodes blocks of Values according to the index for the file.

Updates (writing a newer value for a point that already exists) occur as normal writes.  Since cached values overwrite existing values, newer writes take precedence.

Deletes occur by writing a delete entry for the measurement or series to the WAL and then update the Cache and FileStore.  The Cache evicts all relevant entries.  The FileStore writes a tombstone file for each TSM file that contains relevent data.  These tombstone files are used at startup time to ignore blocks as well as during compactions to remove deleted entries.

# Compactions

Compactions are a serial and continously running process that iteratively optimizes the storage for queries.  Specifically, it does the following:

* Converts closed WAL files into TSM files and removes the closed WAL files
* Combines smaller TSM files into larger ones to improve compression ratios
* Rewrites existing files that contain series data that has been deleted
* Rewrites existing files that contain writes with more recent data to ensure a point exists in only one TSM file.

The compaction algorithm is continously running and always selects files to compact based on a priority.

1. If there are closed WAL files, the 5 oldest WAL segments are added to the set of compaction files.
2. If any TSM files contain points with older timestamps that also exist in the WAL files, those TSM files are added to the compaction set.
3. If any TSM files have a tombstone marker, those TSM files are added to the compaction set.

The compaction is used to generate a set of SeriesIterators that return a sequence of `key`, `Values` where each `key` returned is lexicographically greater than the previous one.  The iterators are ordered such that WAL iterators will override any values return the TSM file iterators.  WAL iterators read and cache the WAL segment so that deletes later in the log can be processed correctly.  TSM file iterators use the tombstone files to ensure that deleted series are not returned during iteration.  As each key is processed, the Values slice is grown, sorted, and then written to a new block in the new TSM file.  The blocks can be split based on number of points or size of the block.  If the total size of the current TSM file would exceed the maximum file size, a new file is created.

Deletions can occur while a new file is being written.  Since the new TSM file is not complete a tombstone would not be written for it. This could result in deleted values getting written into a new file.  To prevent this, if a compaction is running and a delete occurs, the current compaction is aborted and new compaction is started.

When all files are processed and succesfully written, completion checkpoint markers are created and files are renamed.   The engine then notifies the Cache of the checkpoint of the compacted which is used for by the Cache to know what entries can be evicted in the future.

This process then runs again until there are no more WAL files and the minimum number of TSM files exists that are also under the maximum file size.

# WAL

Currently, there is a WAL per shard.  This means all the writes in a WAL segments are for the given shard.  It also means that writes across a lot of shards append to many files which might results in more disk IO due to seeking to the end of multiple files.

Two options being considered:

## WAL per Shard

This is the current behavior of the WAL.  This option is conceptually easier to reason about.  For example, compactions that read in multiple WAL segments are assured that all the WAL entries pertain to the current shard.  If it completes a compaction, it is saft to remove the WAL segment.  It is also easier to deal with shard deletions as all the WAL segments can be dropped along with the other shard files.

The drawback of this option is the potential for turning sequential write IO into random IO in the presence of multiple shards and writes to many different shards.

## Single WAL

Using a single WAL adds some complexity to compactions and deletions.  Compactions will need to either sort all the WAL entries in a segment by shard first and then run compactiosn on each shard or the compactor needs to be able to compact multiple shards concurrently while ensuring points in existing TSM files in different shards remain separate.

Deletions would not be able to reclaim WAL segments immediately as in the case where there is a WAL per shard.  Similarly, a compaction of a WAL segment that contains writes for a deleted shard would need to be dropped.

Currently, we are moving towards a Single WAL implemention.

# Cache

The purpose of the cache is so that data in the WAL is queryable. The client code writes values to the cache, associating a key and checkpoint with each write. The checkpoint must be a monotonically increasing value, but does not have to increase with every write operation. The cache in turn organises all writes first by key (the cache places no constraints on the key as long as it is non-empty) and then by checkpoint. At a time of its choosing, the client also notifies the cache when previously added data has been drained from the WAL. This allows the cache to evict entries associated with all checkpoints up to and including that checkpoint.

The cache tracks its size on a "point-calculated" basis. "Point-calculated" means that the RAM storage footprint for a point in the determined by calling its `Size()` method. While this does not correspond directly to the actual RAM footprint in the cache, the two values are sufficiently correlated for the purpose of controlling RAM.

# TSM File Index

Each TSM file contains a full index of the blocks contained within the file.  The existing index structure is designed to allow for a binary search across the index to find the starting block for a key.  We would then seek to that start key and sequentially scan each block to find the location of a timestamp.

Some issues with the existing structure is that seeking to a given timestamp for a key has a unknown cost.  This can cause variability in read performance that would very difficult to fix.  Another issue is that startup times for loading a TSM file would grow in proportion to number and size of TSM files on disk since we would need to scan the entire file to find all keys contained in the file.  This could be addressed by using a separate index like file or changing the index structure.

We've chosen to update the block index structure to ensure a TSM file is fully self-contained, supports consistent IO characteristics for sequential and random accesses as well as provides an efficient load time regardless of file size.  The implications of these changes are that the index is slightly larger and we need to be able to search the index despite each entry being variably sized.

The following are some alternative design options to handle the cases where the index is too large to fit in memory.  We are currently planning to use an indirect MMAP indexing approach for loaded TSM files.

### Indirect MMAP Indexing

One option is to MMAP the index into memory and record the pointers to the start of each index entry in a slice.  When searching for a given key, the pointers are used to perform a binary search on the underlying mmap data.  When the matching key is found, the block entries can be loaded and search or a subsequent binary search on the blocks can be performed.

A variation of this can also be done without MMAPs by seeking and reading in the file.  The underlying file cache will still be utilized in this approach as well.

As an example, if we have an index structure in memory such as:

 ```
┌────────────────────────────────────────────────────────────────────┐
│                               Index                                │
├─┬──────────────────────┬──┬───────────────────────┬───┬────────────┘
│0│                      │62│                       │145│
├─┴───────┬─────────┬────┼──┴──────┬─────────┬──────┼───┴─────┬──────┐
│Key 1 Len│   Key   │... │Key 2 Len│  Key 2  │ ...  │  Key 3  │ ...  │
│ 2 bytes │ N bytes │    │ 2 bytes │ N bytes │      │ 2 bytes │      │
└─────────┴─────────┴────┴─────────┴─────────┴──────┴─────────┴──────┘
```

We would build an `offsets` slices where each element pointers to the byte location for the first key in then index slice.

```
┌────────────────────────────────────────────────────────────────────┐
│                              Offsets                               │
├────┬────┬────┬─────────────────────────────────────────────────────┘
│ 0  │ 62 │145 │
└────┴────┴────┘
 ```


Using this offset slice we can find `Key 2` by doing a binary search over the offsets slice.  Instead of comparing the value in the offsets (e.g. `62`), we use that as an index into the underlying index to retrieve the key at postion `62` and perform our comparisons with that.

When we have identified the correct position in the index for a given key, we could perform another binary search or a linear scan.  This should be fast as well since each index entry is 28 bytes and all contiguous in memory.

The size of the offsets slice would be proportional to the number of unique series.  If we we limit file sizes to 4GB, we would use 4 bytes for each pointer.

### LRU/Lazy Load

A second option could be to have the index work as a memory bounded, lazy-load style cache.  When a cache miss occurs, the index structure is scanned to find the key and the entries are load and added to the cache which causes the least-recently used entries to be evicted.

### Key Compression

Another option is compress keys using a key specific dictionary encoding.   For example,

```
cpu,host=server1 value=1
cpu,host=server2 value=2
meory,host=server1 value=3
```

Could be compressed by expanding the key into its respective parts: mesasurment, tag keys, tag values and tag fields .  For each part a unique number is assigned.  e.g.

Measurements
```
cpu = 1
memory = 2
```

Tag Keys
```
host = 1
```

Tag Values
```
server1 = 1
server2 = 2
```

Fields
```
value = 1
```

Using this encoding dictionary, the string keys could be converted to a sequency of integers:

```
cpu,host=server1 value=1 -->    1,1,1,1
cpu,host=server2 value=2 -->    1,1,2,1
memory,host=server1 value=3 --> 3,1,2,1
```

These sequences of small integers list can then be compressed further using a bit packed format such as Simple9 or Simple8b.  The resulting byte slices would be a multiple of 4 or 8 bytes (using Simple9/Simple8b respectively) which could used as the (string).

### Separate Index

Another option might be to have a separate index file (BoltDB) that serves as the storage for the `FileIndex` and is transient.   This index would be recreated at startup and updated at compaction time.

# Components

These are some of the high-level components and their responsibilities.  These are ideas preliminary.

## WAL

* Append-only log composed of a fixed size segment files.
* Writes are appended to the current segment
* Roll-over to new segment after filling the current segment
* Closed segments are never modified and used for startup and recovery as well as compactions.
* There is a single WAL for the store as opposed to a WAL per shard.

## Compactor

* Continously running, iterative file storage optimizer
* Takes closed WAL files, existing TSM files and combines into one or more new TSM files

## Cache

* Hold recently written series data
* Has max memory limit
* When limit is crossed, old entries are expired according to the last compaction checkpoint.  Entries written that are older than the last checkpoint time can be evicted.
* If a write comes in, points after the checkpoint are evicted, but there is still not enough room to hold the write, the write returns and error.

# Engine

* Maintains references to Cache, FileStore, WAL, etc..
* Creates a cursor
* Receives writes, coordinates queries
* Hides underlying files and types from clients

## Cursor

* Iterates forward or reverse for given key
* Requests values from Engine for key and timestamp
* Has no knowledge of TSM files or WAL - delegates to Engine to request next set of Values

## FileStore

* Manages TSM files
* Maintains the file indexes and references to active files
* A TSM file that is opened entails reading in and adding the index section to the `FileIndex`.  The block data is then MMAPed up to the index offset to avoid having the index in memory twice.

## FileIndex
* Provides location information to a file and block for a given key and timestamp.

## Interfaces

```
SeriesIterator returns the key and []Value such that a key is only returned
once and subsequent calls to Next() do not return the same key twice.
type SeriesIterator interace {
   func Next() (key, []Value, error)
}
```

## Types

_NOTE: the actual func names are to illustrate the type of functionaltiy the type is responsible._

```
TSMWriter writes a sets of key and Values to a TSM file.
type TSMWriter struct {}
func (t *TSMWriter) Write(key string, values []Value) error {}
func (t *TSMWriter) Close() error
```


```
// WALIterator returns the key and []Values for a set of WAL segment files.
type WALIterator struct{
    Files *os.File
}
func (r *WALReader) Next() (key, []Value, error)
```


```
TSMIterator returns the key and values from a TSM file.
type TSMIterator struct {}
funct (r *TSMIterator) Next() (key, []Value, error)
```

```
type Compactor struct {}
func (c *Compactor) Compact(iters ...SeriesIterators) error
```

```
type Engine struct {
    wal *WAL
    cache *Cache
    fileStore *FileStore
    compactor *Compactor
}

func (e *Engine) ValuesBefore(key string, timstamp time.Time) ([]Value, error)
func (e *Engine) ValuesAfter(key string, timstamp time.Time) ([]Value, error)
```

```
type Cursor struct{
    engine *Engine
}
...
```

```
// FileStore maintains references
type FileStore struct {}
func (f *FileStore) ValuesBefore(key string, timstamp time.Time) ([]Value, error)
func (f *FileStore) ValuesAfter(key string, timstamp time.Time) ([]Value, error)

```

```
type FileIndex struct {}

// Returns a file and offset for a block located in the return file that contains the requested key and timestamp.
func (f *FileIndex) Location(key, timestamp) (*os.File, uint64, error)
```

```
type Cache struct {}
func (c *Cache) Write(key string, values []Value, checkpoint uint64) error
func (c *Cache) SetCheckpoint(checkpoint uint64) error
func (c *Cache) Cursor(key string) tsdb.Cursor
```

```
type WAL struct {}
func (w *WAL) Write(key string, values []Value)
func (w *WAL) ClosedSegments() ([]*os.File, error)
```


# Concerns

## Performance

There are three categories of performance this design is concerned with:

* Write Throughput/Latency
* Query Throughput/Latency
* Startup time
* Compaction Throughput/Latency
* Memory Usage

### Writes

Write throughput is bounded by the time to process the write on the CPU (parsing, sorting, etc..), adding and evicting to the Cache and appending the write to the WAL.  The first two items are CPU bound and can be tuned and optimized if they become a bottleneck.  The WAL write can be tuned such that in the worst case every write requires at least 2 IOPS (write + fysnc) or batched so that multiple writes are queued and fysnc'd in sizes matching one or more disk blocks.  Performing more work with each IO will improve throughput

Write latency is minimal for the WAL write since there are no seeks.  The latency is bounded by the time to complete any write and fysnc calls.

### Queries

Query throughput is directly related to how many blocks can be read in a period of time.  The index structure contains enough information to determine if one or multiple blocks can be read in a single IO.

Query latency is determine by how long it takes to find and read the relevant blocks.  The in-memory index structure contains the offsets and sizes of all blocks for a key.  This allows every block to be read in 2 IOPS (seek + read) regardless of position, structure or size of file.

### Startup

Startup time is proportional to the number of WAL files, TSM files and tombstone files.  WAL files can be read and process in large batches using the WALIterators.  TSM files require reading the index block into memory (5 IOPS/file).  Tombstone files are expected to be small and infrequent and would require approximately 2 IOPS/file.

### Compactions

Compactions are IO intensive in that they may need to read multiple, large TSM files to rewrite them.  The throughput of a compactions (MB/s) as well as the latency for each compaction is important to keep consistent even as data sizes grow.

The performance of compactions also has an effect on what data is visible during queries.  If the Cache fills up and evicts old entries faster than the compactions can process old WAL files, queries could return return gaps until compactions catch up.

To address these concerns, compactions prioritize old WAL files over optimizing storage/compression to avoid data being hidden overload situations.  This also accounts for the fact that shards will eventually become cold for writes so that existing data will be able to be optimized.  To maintain consistent performance, the number of each type of file processed as well as the size of each file processed is bounded.

### Memory Footprint

The memory footprint should shoud not grow unbounded due to additional files or series keys of large sizes or numbers.  Some options for addressing this concern is covered in the [Design Options] section.

## Concurrency

The main concern with concurrency is that reads and writes should not block each other.  Writes add entries to the Cache and append entries to the WAL.  During queries, the contention points will be the Cache and existing TSM files.  Since the Cache and TSM file data is only access through the engine by the cursors, several strategies can be used to improve concurrency.

1. Cache series data can be returned to cursors as a copy.  Since cache entries are evicted on writes, cursors iteration and writes to the same series could block each other.  Iterating over copies of the values can relieve some of this contention.
2. TSM data values returned by the engine are new references to Values and not access to the actual TSM files.  This means that the `Engine`, through the `FileStore` can limit contention.
3. Compactions are the only place where new TSM files are added and removed.  Since this is a serial, continously running process, file contention is minimized.

## Robustness

The two robustness concerns considered by this design are writes filling the cache and crash recovery.

Writes filling up cache faster than the WAL segments can be processed result in the oldest entries being evicted from the cache.  This is the normal operation for the cache.  Old entries are always evicited to make room for new entries.  In the case where WAL segements are slow to be processed, writes are not blocked or errored so timeouts should not occur due to IO issues.  A side effect of this is that queries for recent data will always be served from memory.  The size of the in-memory cache can also be tuned so that if IO does because a bottleneck the window of time for queries with recent data can be tuned.

Crash recovery is handled by using copy-on-write style updates along with checkpoint marker files.  Existing data is never updated.  Updates and deletes to existing data are recored as new changes and processed at compaction and query time.

