// Package index implements encoding and decoding of index format files.
//
//    Git index format
//    ================
//
//    == The Git index file has the following format
//
//      All binary numbers are in network byte order. Version 2 is described
//      here unless stated otherwise.
//
//      - A 12-byte header consisting of
//
//        4-byte signature:
//          The signature is { 'D', 'I', 'R', 'C' } (stands for "dircache")
//
//        4-byte version number:
//          The current supported versions are 2, 3 and 4.
//
//        32-bit number of index entries.
//
//      - A number of sorted index entries (see below).
//
//      - Extensions
//
//        Extensions are identified by signature. Optional extensions can
//        be ignored if Git does not understand them.
//
//        Git currently supports cached tree and resolve undo extensions.
//
//        4-byte extension signature. If the first byte is 'A'..'Z' the
//        extension is optional and can be ignored.
//
//        32-bit size of the extension
//
//        Extension data
//
//      - 160-bit SHA-1 over the content of the index file before this
//        checksum.
//
//    == Index entry
//
//      Index entries are sorted in ascending order on the name field,
//      interpreted as a string of unsigned bytes (i.e. memcmp() order, no
//      localization, no special casing of directory separator '/'). Entries
//      with the same name are sorted by their stage field.
//
//      32-bit ctime seconds, the last time a file's metadata changed
//        this is stat(2) data
//
//      32-bit ctime nanosecond fractions
//        this is stat(2) data
//
//      32-bit mtime seconds, the last time a file's data changed
//        this is stat(2) data
//
//      32-bit mtime nanosecond fractions
//        this is stat(2) data
//
//      32-bit dev
//        this is stat(2) data
//
//      32-bit ino
//        this is stat(2) data
//
//      32-bit mode, split into (high to low bits)
//
//        4-bit object type
//          valid values in binary are 1000 (regular file), 1010 (symbolic link)
//          and 1110 (gitlink)
//
//        3-bit unused
//
//        9-bit unix permission. Only 0755 and 0644 are valid for regular files.
//        Symbolic links and gitlinks have value 0 in this field.
//
//      32-bit uid
//        this is stat(2) data
//
//      32-bit gid
//        this is stat(2) data
//
//      32-bit file size
//        This is the on-disk size from stat(2), truncated to 32-bit.
//
//      160-bit SHA-1 for the represented object
//
//      A 16-bit 'flags' field split into (high to low bits)
//
//        1-bit assume-valid flag
//
//        1-bit extended flag (must be zero in version 2)
//
//        2-bit stage (during merge)
//
//        12-bit name length if the length is less than 0xFFF; otherwise 0xFFF
//        is stored in this field.
//
//      (Version 3 or later) A 16-bit field, only applicable if the
//      "extended flag" above is 1, split into (high to low bits).
//
//        1-bit reserved for future
//
//        1-bit skip-worktree flag (used by sparse checkout)
//
//        1-bit intent-to-add flag (used by "git add -N")
//
//        13-bit unused, must be zero
//
//      Entry path name (variable length) relative to top level directory
//        (without leading slash). '/' is used as path separator. The special
//        path components ".", ".." and ".git" (without quotes) are disallowed.
//        Trailing slash is also disallowed.
//
//        The exact encoding is undefined, but the '.' and '/' characters
//        are encoded in 7-bit ASCII and the encoding cannot contain a NUL
//        byte (iow, this is a UNIX pathname).
//
//      (Version 4) In version 4, the entry path name is prefix-compressed
//        relative to the path name for the previous entry (the very first
//        entry is encoded as if the path name for the previous entry is an
//        empty string).  At the beginning of an entry, an integer N in the
//        variable width encoding (the same encoding as the offset is encoded
//        for OFS_DELTA pack entries; see pack-format.txt) is stored, followed
//        by a NUL-terminated string S.  Removing N bytes from the end of the
//        path name for the previous entry, and replacing it with the string S
//        yields the path name for this entry.
//
//      1-8 nul bytes as necessary to pad the entry to a multiple of eight bytes
//      while keeping the name NUL-terminated.
//
//      (Version 4) In version 4, the padding after the pathname does not
//      exist.
//
//      Interpretation of index entries in split index mode is completely
//      different. See below for details.
//
//    == Extensions
//
//    === Cached tree
//
//      Cached tree extension contains pre-computed hashes for trees that can
//      be derived from the index. It helps speed up tree object generation
//      from index for a new commit.
//
//      When a path is updated in index, the path must be invalidated and
//      removed from tree cache.
//
//      The signature for this extension is { 'T', 'R', 'E', 'E' }.
//
//      A series of entries fill the entire extension; each of which
//      consists of:
//
//      - NUL-terminated path component (relative to its parent directory);
//
//      - ASCII decimal number of entries in the index that is covered by the
//        tree this entry represents (entry_count);
//
//      - A space (ASCII 32);
//
//      - ASCII decimal number that represents the number of subtrees this
//        tree has;
//
//      - A newline (ASCII 10); and
//
//      - 160-bit object name for the object that would result from writing
//        this span of index as a tree.
//
//      An entry can be in an invalidated state and is represented by having
//      a negative number in the entry_count field. In this case, there is no
//      object name and the next entry starts immediately after the newline.
//      When writing an invalid entry, -1 should always be used as entry_count.
//
//      The entries are written out in the top-down, depth-first order.  The
//      first entry represents the root level of the repository, followed by the
//      first subtree--let's call this A--of the root level (with its name
//      relative to the root level), followed by the first subtree of A (with
//      its name relative to A), ...
//
//    === Resolve undo
//
//      A conflict is represented in the index as a set of higher stage entries.
//      When a conflict is resolved (e.g. with "git add path"), these higher
//      stage entries will be removed and a stage-0 entry with proper resolution
//      is added.
//
//      When these higher stage entries are removed, they are saved in the
//      resolve undo extension, so that conflicts can be recreated (e.g. with
//      "git checkout -m"), in case users want to redo a conflict resolution
//      from scratch.
//
//      The signature for this extension is { 'R', 'E', 'U', 'C' }.
//
//      A series of entries fill the entire extension; each of which
//      consists of:
//
//      - NUL-terminated pathname the entry describes (relative to the root of
//        the repository, i.e. full pathname);
//
//      - Three NUL-terminated ASCII octal numbers, entry mode of entries in
//        stage 1 to 3 (a missing stage is represented by "0" in this field);
//        and
//
//      - At most three 160-bit object names of the entry in stages from 1 to 3
//        (nothing is written for a missing stage).
//
//    === Split index
//
//      In split index mode, the majority of index entries could be stored
//      in a separate file. This extension records the changes to be made on
//      top of that to produce the final index.
//
//      The signature for this extension is { 'l', 'i', 'n', 'k' }.
//
//      The extension consists of:
//
//      - 160-bit SHA-1 of the shared index file. The shared index file path
//        is $GIT_DIR/sharedindex.<SHA-1>. If all 160 bits are zero, the
//        index does not require a shared index file.
//
//      - An ewah-encoded delete bitmap, each bit represents an entry in the
//        shared index. If a bit is set, its corresponding entry in the
//        shared index will be removed from the final index.  Note, because
//        a delete operation changes index entry positions, but we do need
//        original positions in replace phase, it's best to just mark
//        entries for removal, then do a mass deletion after replacement.
//
//      - An ewah-encoded replace bitmap, each bit represents an entry in
//        the shared index. If a bit is set, its corresponding entry in the
//        shared index will be replaced with an entry in this index
//        file. All replaced entries are stored in sorted order in this
//        index. The first "1" bit in the replace bitmap corresponds to the
//        first index entry, the second "1" bit to the second entry and so
//        on. Replaced entries may have empty path names to save space.
//
//      The remaining index entries after replaced ones will be added to the
//      final index. These added entries are also sorted by entry name then
//      stage.
//
//    == Untracked cache
//
//      Untracked cache saves the untracked file list and necessary data to
//      verify the cache. The signature for this extension is { 'U', 'N',
//      'T', 'R' }.
//
//      The extension starts with
//
//      - A sequence of NUL-terminated strings, preceded by the size of the
//        sequence in variable width encoding. Each string describes the
//        environment where the cache can be used.
//
//      - Stat data of $GIT_DIR/info/exclude. See "Index entry" section from
//        ctime field until "file size".
//
//      - Stat data of plumbing.excludesfile
//
//      - 32-bit dir_flags (see struct dir_struct)
//
//      - 160-bit SHA-1 of $GIT_DIR/info/exclude. Null SHA-1 means the file
//        does not exist.
//
//      - 160-bit SHA-1 of plumbing.excludesfile. Null SHA-1 means the file does
//        not exist.
//
//      - NUL-terminated string of per-dir exclude file name. This usually
//        is ".gitignore".
//
//      - The number of following directory blocks, variable width
//        encoding. If this number is zero, the extension ends here with a
//        following NUL.
//
//      - A number of directory blocks in depth-first-search order, each
//        consists of
//
//        - The number of untracked entries, variable width encoding.
//
//        - The number of sub-directory blocks, variable width encoding.
//
//        - The directory name terminated by NUL.
//
//        - A number of untracked file/dir names terminated by NUL.
//
//    The remaining data of each directory block is grouped by type:
//
//      - An ewah bitmap, the n-th bit marks whether the n-th directory has
//        valid untracked cache entries.
//
//      - An ewah bitmap, the n-th bit records "check-only" bit of
//        read_directory_recursive() for the n-th directory.
//
//      - An ewah bitmap, the n-th bit indicates whether SHA-1 and stat data
//        is valid for the n-th directory and exists in the next data.
//
//      - An array of stat data. The n-th data corresponds with the n-th
//        "one" bit in the previous ewah bitmap.
//
//      - An array of SHA-1. The n-th SHA-1 corresponds with the n-th "one" bit
//        in the previous ewah bitmap.
//
//      - One NUL.
// Source https://www.kernel.org/pub/software/scm/git/docs/technical/index-format.txt
package index
