package devmapper

// Definition of struct dm_task and sub structures (from lvm2)
//
// struct dm_ioctl {
// 	/*
// 	 * The version number is made up of three parts:
// 	 * major - no backward or forward compatibility,
// 	 * minor - only backwards compatible,
// 	 * patch - both backwards and forwards compatible.
// 	 *
// 	 * All clients of the ioctl interface should fill in the
// 	 * version number of the interface that they were
// 	 * compiled with.
// 	 *
// 	 * All recognized ioctl commands (ie. those that don't
// 	 * return -ENOTTY) fill out this field, even if the
// 	 * command failed.
// 	 */
// 	uint32_t version[3];	/* in/out */
// 	uint32_t data_size;	/* total size of data passed in
// 				 * including this struct */

// 	uint32_t data_start;	/* offset to start of data
// 				 * relative to start of this struct */

// 	uint32_t target_count;	/* in/out */
// 	int32_t open_count;	/* out */
// 	uint32_t flags;		/* in/out */

// 	/*
// 	 * event_nr holds either the event number (input and output) or the
// 	 * udev cookie value (input only).
// 	 * The DM_DEV_WAIT ioctl takes an event number as input.
// 	 * The DM_SUSPEND, DM_DEV_REMOVE and DM_DEV_RENAME ioctls
// 	 * use the field as a cookie to return in the DM_COOKIE
// 	 * variable with the uevents they issue.
// 	 * For output, the ioctls return the event number, not the cookie.
// 	 */
// 	uint32_t event_nr;      	/* in/out */
// 	uint32_t padding;

// 	uint64_t dev;		/* in/out */

// 	char name[DM_NAME_LEN];	/* device name */
// 	char uuid[DM_UUID_LEN];	/* unique identifier for
// 				 * the block device */
// 	char data[7];		/* padding or data */
// };

// struct target {
// 	uint64_t start;
// 	uint64_t length;
// 	char *type;
// 	char *params;

// 	struct target *next;
// };

// typedef enum {
// 	DM_ADD_NODE_ON_RESUME, /* add /dev/mapper node with dmsetup resume */
// 	DM_ADD_NODE_ON_CREATE  /* add /dev/mapper node with dmsetup create */
// } dm_add_node_t;

// struct dm_task {
// 	int type;
// 	char *dev_name;
// 	char *mangled_dev_name;

// 	struct target *head, *tail;

// 	int read_only;
// 	uint32_t event_nr;
// 	int major;
// 	int minor;
// 	int allow_default_major_fallback;
// 	uid_t uid;
// 	gid_t gid;
// 	mode_t mode;
// 	uint32_t read_ahead;
// 	uint32_t read_ahead_flags;
// 	union {
// 		struct dm_ioctl *v4;
// 	} dmi;
// 	char *newname;
// 	char *message;
// 	char *geometry;
// 	uint64_t sector;
// 	int no_flush;
// 	int no_open_count;
// 	int skip_lockfs;
// 	int query_inactive_table;
// 	int suppress_identical_reload;
// 	dm_add_node_t add_node;
// 	uint64_t existing_table_size;
// 	int cookie_set;
// 	int new_uuid;
// 	int secure_data;
// 	int retry_remove;
// 	int enable_checks;
// 	int expected_errno;

// 	char *uuid;
// 	char *mangled_uuid;
// };
//
