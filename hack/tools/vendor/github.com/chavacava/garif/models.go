package garif

// Address A physical or virtual address, or a range of addresses, in an 'addressable region' (memory or a binary file).
type Address struct {

	// The address expressed as a byte offset from the start of the addressable region.
	AbsoluteAddress int `json:"absoluteAddress,omitempty"`

	// A human-readable fully qualified name that is associated with the address.
	FullyQualifiedName string `json:"fullyQualifiedName,omitempty"`

	// The index within run.addresses of the cached object for this address.
	Index int `json:"index,omitempty"`

	// An open-ended string that identifies the address kind.
	// 'data', 'function', 'header','instruction', 'module', 'page', 'section',
	// 'segment', 'stack', 'stackFrame', 'table' are well-known values.
	Kind string `json:"kind,omitempty"`

	// The number of bytes in this range of addresses.
	Length int `json:"length,omitempty"`

	// A name that is associated with the address, e.g., '.text'.
	Name string `json:"name,omitempty"`

	// The byte offset of this address from the absolute or relative address of the parent object.
	OffsetFromParent int `json:"offsetFromParent,omitempty"`

	// The index within run.addresses of the parent object.
	ParentIndex int `json:"parentIndex,omitempty"`

	// Key/value pairs that provide additional information about the address.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The address expressed as a byte offset from the absolute address of the top-most parent object.
	RelativeAddress int `json:"relativeAddress,omitempty"`
}

// Artifact A single artifact. In some cases, this artifact might be nested within another artifact.
type Artifact struct {

	// The contents of the artifact.
	Contents *ArtifactContent `json:"contents,omitempty"`

	// A short description of the artifact.
	Description *Message `json:"description,omitempty"`

	// Specifies the encoding for an artifact object that refers to a text file.
	Encoding string `json:"encoding,omitempty"`

	// A dictionary, each of whose keys is the name of a hash function and each of whose values is
	// the hashed value of the artifact produced by the specified hash function.
	Hashes map[string]string `json:"hashes,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which the artifact was most recently modified.
	// See "Date/time properties" in the SARIF spec for the required format.
	LastModifiedTimeUtc string `json:"lastModifiedTimeUtc,omitempty"`

	// The length of the artifact in bytes.
	Length int `json:"length,omitempty"`

	// The location of the artifact.
	Location *ArtifactLocation `json:"location,omitempty"`

	// The MIME type (RFC 2045) of the artifact.
	MimeType string `json:"mimeType,omitempty"`

	// The offset in bytes of the artifact within its containing artifact.
	Offset int `json:"offset,omitempty"`

	// Identifies the index of the immediate parent of the artifact, if this artifact is nested.
	ParentIndex int `json:"parentIndex,omitempty"`

	// Key/value pairs that provide additional information about the artifact.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The role or roles played by the artifact in the analysis.
	Roles []interface{} `json:"roles,omitempty"`

	// Specifies the source language for any artifact object that refers to a text file that contains source code.
	SourceLanguage string `json:"sourceLanguage,omitempty"`
}

// ArtifactChange A change to a single artifact.
type ArtifactChange struct {

	// The location of the artifact to change.
	ArtifactLocation *ArtifactLocation `json:"artifactLocation"`

	// Key/value pairs that provide additional information about the change.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of replacement objects, each of which represents the replacement of a single region in a
	// single artifact specified by 'artifactLocation'.
	Replacements []*Replacement `json:"replacements"`
}

// ArtifactContent Represents the contents of an artifact.
type ArtifactContent struct {

	// MIME Base64-encoded content from a binary artifact, or from a text artifact in its original encoding.
	Binary string `json:"binary,omitempty"`

	// Key/value pairs that provide additional information about the artifact content.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An alternate rendered representation of the artifact (e.g., a decompiled representation of a binary region).
	Rendered *MultiformatMessageString `json:"rendered,omitempty"`

	// UTF-8-encoded content from a text artifact.
	Text string `json:"text,omitempty"`
}

// ArtifactLocation Specifies the location of an artifact.
type ArtifactLocation struct {

	// A short description of the artifact location.
	Description *Message `json:"description,omitempty"`

	// The index within the run artifacts array of the artifact object associated with the artifact location.
	Index int `json:"index,omitempty"`

	// Key/value pairs that provide additional information about the artifact location.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A string containing a valid relative or absolute URI.
	Uri string `json:"uri,omitempty"`

	// A string which indirectly specifies the absolute URI with respect to which a relative URI in the "uri" property is interpreted.
	UriBaseId string `json:"uriBaseId,omitempty"`
}

// Attachment An artifact relevant to a result.
type Attachment struct {

	// The location of the attachment.
	ArtifactLocation *ArtifactLocation `json:"artifactLocation"`

	// A message describing the role played by the attachment.
	Description *Message `json:"description,omitempty"`

	// Key/value pairs that provide additional information about the attachment.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of rectangles specifying areas of interest within the image.
	Rectangles []*Rectangle `json:"rectangles,omitempty"`

	// An array of regions of interest within the attachment.
	Regions []*Region `json:"regions,omitempty"`
}

// CodeFlow A set of threadFlows which together describe a pattern of code execution relevant to detecting a result.
type CodeFlow struct {

	// A message relevant to the code flow.
	Message *Message `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the code flow.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of one or more unique threadFlow objects, each of which describes the progress of a program
	// through a thread of execution.
	ThreadFlows []*ThreadFlow `json:"threadFlows"`
}

// ConfigurationOverride Information about how a specific rule or notification was reconfigured at runtime.
type ConfigurationOverride struct {

	// Specifies how the rule or notification was configured during the scan.
	Configuration *ReportingConfiguration `json:"configuration"`

	// A reference used to locate the descriptor whose configuration was overridden.
	Descriptor *ReportingDescriptorReference `json:"descriptor"`

	// Key/value pairs that provide additional information about the configuration override.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// Conversion Describes how a converter transformed the output of a static analysis tool from the analysis tool's native output format into the SARIF format.
type Conversion struct {

	// The locations of the analysis tool's per-run log files.
	AnalysisToolLogFiles []*ArtifactLocation `json:"analysisToolLogFiles,omitempty"`

	// An invocation object that describes the invocation of the converter.
	Invocation *Invocation `json:"invocation,omitempty"`

	// Key/value pairs that provide additional information about the conversion.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A tool object that describes the converter.
	Tool *Tool `json:"tool"`
}

// Edge Represents a directed edge in a graph.
type Edge struct {

	// A string that uniquely identifies the edge within its graph.
	Id string `json:"id"`

	// A short description of the edge.
	Label *Message `json:"label,omitempty"`

	// Key/value pairs that provide additional information about the edge.
	Properties *PropertyBag `json:"properties,omitempty"`

	// Identifies the source node (the node at which the edge starts).
	SourceNodeId string `json:"sourceNodeId"`

	// Identifies the target node (the node at which the edge ends).
	TargetNodeId string `json:"targetNodeId"`
}

// EdgeTraversal Represents the traversal of a single edge during a graph traversal.
type EdgeTraversal struct {

	// Identifies the edge being traversed.
	EdgeId string `json:"edgeId"`

	// The values of relevant expressions after the edge has been traversed.
	FinalState map[string]*MultiformatMessageString `json:"finalState,omitempty"`

	// A message to display to the user as the edge is traversed.
	Message *Message `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the edge traversal.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The number of edge traversals necessary to return from a nested graph.
	StepOverEdgeCount int `json:"stepOverEdgeCount,omitempty"`
}

// Exception Describes a runtime exception encountered during the execution of an analysis tool.
type Exception struct {

	// An array of exception objects each of which is considered a cause of this exception.
	InnerExceptions []*Exception `json:"innerExceptions,omitempty"`

	// A string that identifies the kind of exception, for example, the fully qualified type name of an object that was thrown, or the symbolic name of a signal.
	Kind string `json:"kind,omitempty"`

	// A message that describes the exception.
	Message string `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the exception.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The sequence of function calls leading to the exception.
	Stack *Stack `json:"stack,omitempty"`
}

// ExternalProperties The top-level element of an external property file.
type ExternalProperties struct {

	// Addresses that will be merged with a separate run.
	Addresses []*Address `json:"addresses,omitempty"`

	// An array of artifact objects that will be merged with a separate run.
	Artifacts []*Artifact `json:"artifacts,omitempty"`

	// A conversion object that will be merged with a separate run.
	Conversion *Conversion `json:"conversion,omitempty"`

	// The analysis tool object that will be merged with a separate run.
	Driver *ToolComponent `json:"driver,omitempty"`

	// Tool extensions that will be merged with a separate run.
	Extensions []*ToolComponent `json:"extensions,omitempty"`

	// Key/value pairs that provide additional information that will be merged with a separate run.
	ExternalizedProperties *PropertyBag `json:"externalizedProperties,omitempty"`

	// An array of graph objects that will be merged with a separate run.
	Graphs []*Graph `json:"graphs,omitempty"`

	// A stable, unique identifer for this external properties object, in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// Describes the invocation of the analysis tool that will be merged with a separate run.
	Invocations []*Invocation `json:"invocations,omitempty"`

	// An array of logical locations such as namespaces, types or functions that will be merged with a separate run.
	LogicalLocations []*LogicalLocation `json:"logicalLocations,omitempty"`

	// Tool policies that will be merged with a separate run.
	Policies []*ToolComponent `json:"policies,omitempty"`

	// Key/value pairs that provide additional information about the external properties.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of result objects that will be merged with a separate run.
	Results []*Result `json:"results,omitempty"`

	// A stable, unique identifer for the run associated with this external properties object, in the form of a GUID.
	RunGuid string `json:"runGuid,omitempty"`

	// The URI of the JSON schema corresponding to the version of the external property file format.
	Schema string `json:"schema,omitempty"`

	// Tool taxonomies that will be merged with a separate run.
	Taxonomies []*ToolComponent `json:"taxonomies,omitempty"`

	// An array of threadFlowLocation objects that will be merged with a separate run.
	ThreadFlowLocations []*ThreadFlowLocation `json:"threadFlowLocations,omitempty"`

	// Tool translations that will be merged with a separate run.
	Translations []*ToolComponent `json:"translations,omitempty"`

	// The SARIF format version of this external properties object.
	Version interface{} `json:"version,omitempty"`

	// Requests that will be merged with a separate run.
	WebRequests []*WebRequest `json:"webRequests,omitempty"`

	// Responses that will be merged with a separate run.
	WebResponses []*WebResponse `json:"webResponses,omitempty"`
}

// ExternalPropertyFileReference Contains information that enables a SARIF consumer to locate the external property file that contains the value of an externalized property associated with the run.
type ExternalPropertyFileReference struct {

	// A stable, unique identifer for the external property file in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// A non-negative integer specifying the number of items contained in the external property file.
	ItemCount int `json:"itemCount,omitempty"`

	// The location of the external property file.
	Location *ArtifactLocation `json:"location,omitempty"`

	// Key/value pairs that provide additional information about the external property file.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// ExternalPropertyFileReferences References to external property files that should be inlined with the content of a root log file.
type ExternalPropertyFileReferences struct {

	// An array of external property files containing run.addresses arrays to be merged with the root log file.
	Addresses []*ExternalPropertyFileReference `json:"addresses,omitempty"`

	// An array of external property files containing run.artifacts arrays to be merged with the root log file.
	Artifacts []*ExternalPropertyFileReference `json:"artifacts,omitempty"`

	// An external property file containing a run.conversion object to be merged with the root log file.
	Conversion *ExternalPropertyFileReference `json:"conversion,omitempty"`

	// An external property file containing a run.driver object to be merged with the root log file.
	Driver *ExternalPropertyFileReference `json:"driver,omitempty"`

	// An array of external property files containing run.extensions arrays to be merged with the root log file.
	Extensions []*ExternalPropertyFileReference `json:"extensions,omitempty"`

	// An external property file containing a run.properties object to be merged with the root log file.
	ExternalizedProperties *ExternalPropertyFileReference `json:"externalizedProperties,omitempty"`

	// An array of external property files containing a run.graphs object to be merged with the root log file.
	Graphs []*ExternalPropertyFileReference `json:"graphs,omitempty"`

	// An array of external property files containing run.invocations arrays to be merged with the root log file.
	Invocations []*ExternalPropertyFileReference `json:"invocations,omitempty"`

	// An array of external property files containing run.logicalLocations arrays to be merged with the root log file.
	LogicalLocations []*ExternalPropertyFileReference `json:"logicalLocations,omitempty"`

	// An array of external property files containing run.policies arrays to be merged with the root log file.
	Policies []*ExternalPropertyFileReference `json:"policies,omitempty"`

	// Key/value pairs that provide additional information about the external property files.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of external property files containing run.results arrays to be merged with the root log file.
	Results []*ExternalPropertyFileReference `json:"results,omitempty"`

	// An array of external property files containing run.taxonomies arrays to be merged with the root log file.
	Taxonomies []*ExternalPropertyFileReference `json:"taxonomies,omitempty"`

	// An array of external property files containing run.threadFlowLocations arrays to be merged with the root log file.
	ThreadFlowLocations []*ExternalPropertyFileReference `json:"threadFlowLocations,omitempty"`

	// An array of external property files containing run.translations arrays to be merged with the root log file.
	Translations []*ExternalPropertyFileReference `json:"translations,omitempty"`

	// An array of external property files containing run.requests arrays to be merged with the root log file.
	WebRequests []*ExternalPropertyFileReference `json:"webRequests,omitempty"`

	// An array of external property files containing run.responses arrays to be merged with the root log file.
	WebResponses []*ExternalPropertyFileReference `json:"webResponses,omitempty"`
}

// Fix A proposed fix for the problem represented by a result object.
// A fix specifies a set of artifacts to modify. For each artifact,
// it specifies a set of bytes to remove, and provides a set of new bytes to replace them.
type Fix struct {

	// One or more artifact changes that comprise a fix for a result.
	ArtifactChanges []*ArtifactChange `json:"artifactChanges"`

	// A message that describes the proposed fix, enabling viewers to present the proposed change to an end user.
	Description *Message `json:"description,omitempty"`

	// Key/value pairs that provide additional information about the fix.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// Graph A network of nodes and directed edges that describes some aspect of the
// structure of the code (for example, a call graph).
type Graph struct {

	// A description of the graph.
	Description *Message `json:"description,omitempty"`

	// An array of edge objects representing the edges of the graph.
	Edges []*Edge `json:"edges,omitempty"`

	// An array of node objects representing the nodes of the graph.
	Nodes []*Node `json:"nodes,omitempty"`

	// Key/value pairs that provide additional information about the graph.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// GraphTraversal Represents a path through a graph.
type GraphTraversal struct {

	// A description of this graph traversal.
	Description *Message `json:"description,omitempty"`

	// The sequences of edges traversed by this graph traversal.
	EdgeTraversals []*EdgeTraversal `json:"edgeTraversals,omitempty"`

	// Values of relevant expressions at the start of the graph traversal that remain constant for the graph traversal.
	ImmutableState map[string]*MultiformatMessageString `json:"immutableState,omitempty"`

	// Values of relevant expressions at the start of the graph traversal that may change during graph traversal.
	InitialState map[string]*MultiformatMessageString `json:"initialState,omitempty"`

	// Key/value pairs that provide additional information about the graph traversal.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The index within the result.graphs to be associated with the result.
	ResultGraphIndex int `json:"resultGraphIndex,omitempty"`

	// The index within the run.graphs to be associated with the result.
	RunGraphIndex int `json:"runGraphIndex,omitempty"`
}

// Invocation The runtime environment of the analysis tool run.
type Invocation struct {

	// The account under which the invocation occurred.
	Account string `json:"account,omitempty"`

	// An array of strings, containing in order the command line arguments passed to the tool from the operating system.
	Arguments []string `json:"arguments,omitempty"`

	// The command line used to invoke the tool.
	CommandLine string `json:"commandLine,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which the invocation ended. See "Date/time properties" in the SARIF spec for the required format.
	EndTimeUtc string `json:"endTimeUtc,omitempty"`

	// The environment variables associated with the analysis tool process, expressed as key/value pairs.
	EnvironmentVariables map[string]string `json:"environmentVariables,omitempty"`

	// An absolute URI specifying the location of the executable that was invoked.
	ExecutableLocation *ArtifactLocation `json:"executableLocation,omitempty"`

	// Specifies whether the tool's execution completed successfully.
	ExecutionSuccessful bool `json:"executionSuccessful"`

	// The process exit code.
	ExitCode int `json:"exitCode,omitempty"`

	// The reason for the process exit.
	ExitCodeDescription string `json:"exitCodeDescription,omitempty"`

	// The name of the signal that caused the process to exit.
	ExitSignalName string `json:"exitSignalName,omitempty"`

	// The numeric value of the signal that caused the process to exit.
	ExitSignalNumber int `json:"exitSignalNumber,omitempty"`

	// The machine on which the invocation occurred.
	Machine string `json:"machine,omitempty"`

	// An array of configurationOverride objects that describe notifications related runtime overrides.
	NotificationConfigurationOverrides []*ConfigurationOverride `json:"notificationConfigurationOverrides,omitempty"`

	// The id of the process in which the invocation occurred.
	ProcessId int `json:"processId,omitempty"`

	// The reason given by the operating system that the process failed to start.
	ProcessStartFailureMessage string `json:"processStartFailureMessage,omitempty"`

	// Key/value pairs that provide additional information about the invocation.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The locations of any response files specified on the tool's command line.
	ResponseFiles []*ArtifactLocation `json:"responseFiles,omitempty"`

	// An array of configurationOverride objects that describe rules related runtime overrides.
	RuleConfigurationOverrides []*ConfigurationOverride `json:"ruleConfigurationOverrides,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which the invocation started. See "Date/time properties" in the SARIF spec for the required format.
	StartTimeUtc string `json:"startTimeUtc,omitempty"`

	// A file containing the standard error stream from the process that was invoked.
	Stderr *ArtifactLocation `json:"stderr,omitempty"`

	// A file containing the standard input stream to the process that was invoked.
	Stdin *ArtifactLocation `json:"stdin,omitempty"`

	// A file containing the standard output stream from the process that was invoked.
	Stdout *ArtifactLocation `json:"stdout,omitempty"`

	// A file containing the interleaved standard output and standard error stream from the process that was invoked.
	StdoutStderr *ArtifactLocation `json:"stdoutStderr,omitempty"`

	// A list of conditions detected by the tool that are relevant to the tool's configuration.
	ToolConfigurationNotifications []*Notification `json:"toolConfigurationNotifications,omitempty"`

	// A list of runtime conditions detected by the tool during the analysis.
	ToolExecutionNotifications []*Notification `json:"toolExecutionNotifications,omitempty"`

	// The working directory for the invocation.
	WorkingDirectory *ArtifactLocation `json:"workingDirectory,omitempty"`
}

// Location A location within a programming artifact.
type Location struct {

	// A set of regions relevant to the location.
	Annotations []*Region `json:"annotations,omitempty"`

	// Value that distinguishes this location from all other locations within a single result object.
	Id int `json:"id,omitempty"`

	// The logical locations associated with the result.
	LogicalLocations []*LogicalLocation `json:"logicalLocations,omitempty"`

	// A message relevant to the location.
	Message *Message `json:"message,omitempty"`

	// Identifies the artifact and region.
	PhysicalLocation *PhysicalLocation `json:"physicalLocation,omitempty"`

	// Key/value pairs that provide additional information about the location.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of objects that describe relationships between this location and others.
	Relationships []*LocationRelationship `json:"relationships,omitempty"`
}

// LocationRelationship Information about the relation of one location to another.
type LocationRelationship struct {

	// A description of the location relationship.
	Description *Message `json:"description,omitempty"`

	// A set of distinct strings that categorize the relationship. Well-known kinds include 'includes', 'isIncludedBy' and 'relevant'.
	Kinds []string `json:"kinds,omitempty"`

	// Key/value pairs that provide additional information about the location relationship.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A reference to the related location.
	Target int `json:"target"`
}

// LogFile Static Analysis Results Format (SARIF) Version 2.1.0 JSON Schema.
type LogFile struct {

	// References to external property files that share data between runs.
	InlineExternalProperties []*ExternalProperties `json:"inlineExternalProperties,omitempty"`

	// Key/value pairs that provide additional information about the log file.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The set of runs contained in this log file.
	Runs []*Run `json:"runs"`

	// The URI of the JSON schema corresponding to the version.
	Schema string `json:"$schema,omitempty"`

	// The SARIF format version of this log file.
	Version interface{} `json:"version"`
}

// LogicalLocation A logical location of a construct that produced a result.
type LogicalLocation struct {

	// The machine-readable name for the logical location, such as a mangled function name provided by a C++ compiler that encodes calling convention, return type and other details along with the function name.
	DecoratedName string `json:"decoratedName,omitempty"`

	// The human-readable fully qualified name of the logical location.
	FullyQualifiedName string `json:"fullyQualifiedName,omitempty"`

	// The index within the logical locations array.
	Index int `json:"index,omitempty"`

	// The type of construct this logical location component refers to. Should be one of 'function', 'member', 'module', 'namespace', 'parameter', 'resource', 'returnType', 'type', 'variable', 'object', 'array', 'property', 'value', 'element', 'text', 'attribute', 'comment', 'declaration', 'dtd' or 'processingInstruction', if any of those accurately describe the construct.
	Kind string `json:"kind,omitempty"`

	// Identifies the construct in which the result occurred. For example, this property might contain the name of a class or a method.
	Name string `json:"name,omitempty"`

	// Identifies the index of the immediate parent of the construct in which the result was detected. For example, this property might point to a logical location that represents the namespace that holds a type.
	ParentIndex int `json:"parentIndex,omitempty"`

	// Key/value pairs that provide additional information about the logical location.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// Message Encapsulates a message intended to be read by the end user.
type Message struct {

	// An array of strings to substitute into the message string.
	Arguments []string `json:"arguments,omitempty"`

	// The identifier for this message.
	Id string `json:"id,omitempty"`

	// A Markdown message string.
	Markdown string `json:"markdown,omitempty"`

	// Key/value pairs that provide additional information about the message.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A plain text message string.
	Text string `json:"text,omitempty"`
}

// MultiformatMessageString A message string or message format string rendered in multiple formats.
type MultiformatMessageString struct {

	// A Markdown message string or format string.
	Markdown string `json:"markdown,omitempty"`

	// Key/value pairs that provide additional information about the message.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A plain text message string or format string.
	Text string `json:"text"`
}

// Node Represents a node in a graph.
type Node struct {

	// Array of child nodes.
	Children []*Node `json:"children,omitempty"`

	// A string that uniquely identifies the node within its graph.
	Id string `json:"id"`

	// A short description of the node.
	Label *Message `json:"label,omitempty"`

	// A code location associated with the node.
	Location *Location `json:"location,omitempty"`

	// Key/value pairs that provide additional information about the node.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// Notification Describes a condition relevant to the tool itself, as opposed to being relevant to a target being analyzed by the tool.
type Notification struct {

	// A reference used to locate the rule descriptor associated with this notification.
	AssociatedRule *ReportingDescriptorReference `json:"associatedRule,omitempty"`

	// A reference used to locate the descriptor relevant to this notification.
	Descriptor *ReportingDescriptorReference `json:"descriptor,omitempty"`

	// The runtime exception, if any, relevant to this notification.
	Exception *Exception `json:"exception,omitempty"`

	// A value specifying the severity level of the notification.
	Level interface{} `json:"level,omitempty"`

	// The locations relevant to this notification.
	Locations []*Location `json:"locations,omitempty"`

	// A message that describes the condition that was encountered.
	Message *Message `json:"message"`

	// Key/value pairs that provide additional information about the notification.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The thread identifier of the code that generated the notification.
	ThreadId int `json:"threadId,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which the analysis tool generated the notification.
	TimeUtc string `json:"timeUtc,omitempty"`
}

// PhysicalLocation A physical location relevant to a result. Specifies a reference to a programming artifact together with a range of bytes or characters within that artifact.
type PhysicalLocation struct {

	// The address of the location.
	Address *Address `json:"address,omitempty"`

	// The location of the artifact.
	ArtifactLocation *ArtifactLocation `json:"artifactLocation,omitempty"`

	// Specifies a portion of the artifact that encloses the region. Allows a viewer to display additional context around the region.
	ContextRegion *Region `json:"contextRegion,omitempty"`

	// Key/value pairs that provide additional information about the physical location.
	Properties *PropertyBag `json:"properties,omitempty"`

	// Specifies a portion of the artifact.
	Region *Region `json:"region,omitempty"`
}

type PropertyBag map[string]interface{}

/*
// PropertyBag Key/value pairs that provide additional information about the object.
type PropertyBag struct {
	AdditionalProperties map[string]interface{} `json:"-,omitempty"`

	// A set of distinct strings that provide additional information.
	Tags []string `json:"tags,omitempty"`
}
*/
// Rectangle An area within an image.
type Rectangle struct {

	// The Y coordinate of the bottom edge of the rectangle, measured in the image's natural units.
	Bottom float64 `json:"bottom,omitempty"`

	// The X coordinate of the left edge of the rectangle, measured in the image's natural units.
	Left float64 `json:"left,omitempty"`

	// A message relevant to the rectangle.
	Message *Message `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the rectangle.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The X coordinate of the right edge of the rectangle, measured in the image's natural units.
	Right float64 `json:"right,omitempty"`

	// The Y coordinate of the top edge of the rectangle, measured in the image's natural units.
	Top float64 `json:"top,omitempty"`
}

// Region A region within an artifact where a result was detected.
type Region struct {

	// The length of the region in bytes.
	ByteLength int `json:"byteLength,omitempty"`

	// The zero-based offset from the beginning of the artifact of the first byte in the region.
	ByteOffset int `json:"byteOffset,omitempty"`

	// The length of the region in characters.
	CharLength int `json:"charLength,omitempty"`

	// The zero-based offset from the beginning of the artifact of the first character in the region.
	CharOffset int `json:"charOffset,omitempty"`

	// The column number of the character following the end of the region.
	EndColumn int `json:"endColumn,omitempty"`

	// The line number of the last character in the region.
	EndLine int `json:"endLine,omitempty"`

	// A message relevant to the region.
	Message *Message `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the region.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The portion of the artifact contents within the specified region.
	Snippet *ArtifactContent `json:"snippet,omitempty"`

	// Specifies the source language, if any, of the portion of the artifact specified by the region object.
	SourceLanguage string `json:"sourceLanguage,omitempty"`

	// The column number of the first character in the region.
	StartColumn int `json:"startColumn,omitempty"`

	// The line number of the first character in the region.
	StartLine int `json:"startLine,omitempty"`
}

// Replacement The replacement of a single region of an artifact.
type Replacement struct {

	// The region of the artifact to delete.
	DeletedRegion *Region `json:"deletedRegion"`

	// The content to insert at the location specified by the 'deletedRegion' property.
	InsertedContent *ArtifactContent `json:"insertedContent,omitempty"`

	// Key/value pairs that provide additional information about the replacement.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// ReportingConfiguration Information about a rule or notification that can be configured at runtime.
type ReportingConfiguration struct {

	// Specifies whether the report may be produced during the scan.
	Enabled bool `json:"enabled,omitempty"`

	// Specifies the failure level for the report.
	Level interface{} `json:"level,omitempty"`

	// Contains configuration information specific to a report.
	Parameters *PropertyBag `json:"parameters,omitempty"`

	// Key/value pairs that provide additional information about the reporting configuration.
	Properties *PropertyBag `json:"properties,omitempty"`

	// Specifies the relative priority of the report. Used for analysis output only.
	Rank float64 `json:"rank,omitempty"`
}

// ReportingDescriptor Metadata that describes a specific report produced by the tool, as part of the analysis it provides or its runtime reporting.
type ReportingDescriptor struct {

	// Default reporting configuration information.
	DefaultConfiguration *ReportingConfiguration `json:"defaultConfiguration,omitempty"`

	// An array of unique identifies in the form of a GUID by which this report was known in some previous version of the analysis tool.
	DeprecatedGuids []string `json:"deprecatedGuids,omitempty"`

	// An array of stable, opaque identifiers by which this report was known in some previous version of the analysis tool.
	DeprecatedIds []string `json:"deprecatedIds,omitempty"`

	// An array of readable identifiers by which this report was known in some previous version of the analysis tool.
	DeprecatedNames []string `json:"deprecatedNames,omitempty"`

	// A description of the report. Should, as far as possible, provide details sufficient to enable resolution of any problem indicated by the result.
	FullDescription *MultiformatMessageString `json:"fullDescription,omitempty"`

	// A unique identifer for the reporting descriptor in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// Provides the primary documentation for the report, useful when there is no online documentation.
	Help *MultiformatMessageString `json:"help,omitempty"`

	// A URI where the primary documentation for the report can be found.
	HelpUri string `json:"helpUri,omitempty"`

	// A stable, opaque identifier for the report.
	Id string `json:"id"`

	// A set of name/value pairs with arbitrary names. Each value is a multiformatMessageString object, which holds message strings in plain text and (optionally) Markdown format. The strings can include placeholders, which can be used to construct a message in combination with an arbitrary number of additional string arguments.
	MessageStrings map[string]*MultiformatMessageString `json:"messageStrings,omitempty"`

	// A report identifier that is understandable to an end user.
	Name string `json:"name,omitempty"`

	// Key/value pairs that provide additional information about the report.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of objects that describe relationships between this reporting descriptor and others.
	Relationships []*ReportingDescriptorRelationship `json:"relationships,omitempty"`

	// A concise description of the report. Should be a single sentence that is understandable when visible space is limited to a single line of text.
	ShortDescription *MultiformatMessageString `json:"shortDescription,omitempty"`
}

// ReportingDescriptorReference Information about how to locate a relevant reporting descriptor.
type ReportingDescriptorReference struct {

	// A guid that uniquely identifies the descriptor.
	Guid string `json:"guid,omitempty"`

	// The id of the descriptor.
	Id string `json:"id,omitempty"`

	// The index into an array of descriptors in toolComponent.ruleDescriptors, toolComponent.notificationDescriptors, or toolComponent.taxonomyDescriptors, depending on context.
	Index int `json:"index,omitempty"`

	// Key/value pairs that provide additional information about the reporting descriptor reference.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A reference used to locate the toolComponent associated with the descriptor.
	ToolComponent *ToolComponentReference `json:"toolComponent,omitempty"`
}

// ReportingDescriptorRelationship Information about the relation of one reporting descriptor to another.
type ReportingDescriptorRelationship struct {

	// A description of the reporting descriptor relationship.
	Description *Message `json:"description,omitempty"`

	// A set of distinct strings that categorize the relationship. Well-known kinds include 'canPrecede', 'canFollow', 'willPrecede', 'willFollow', 'superset', 'subset', 'equal', 'disjoint', 'relevant', and 'incomparable'.
	Kinds []string `json:"kinds,omitempty"`

	// Key/value pairs that provide additional information about the reporting descriptor reference.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A reference to the related reporting descriptor.
	Target *ReportingDescriptorReference `json:"target"`
}

// Result A result produced by an analysis tool.
type Result struct {

	// Identifies the artifact that the analysis tool was instructed to scan. This need not be the same as the artifact where the result actually occurred.
	AnalysisTarget *ArtifactLocation `json:"analysisTarget,omitempty"`

	// A set of artifacts relevant to the result.
	Attachments []*Attachment `json:"attachments,omitempty"`

	// The state of a result relative to a baseline of a previous run.
	BaselineState interface{} `json:"baselineState,omitempty"`

	// An array of 'codeFlow' objects relevant to the result.
	CodeFlows []*CodeFlow `json:"codeFlows,omitempty"`

	// A stable, unique identifier for the equivalence class of logically identical results to which this result belongs, in the form of a GUID.
	CorrelationGuid string `json:"correlationGuid,omitempty"`

	// A set of strings each of which individually defines a stable, unique identity for the result.
	Fingerprints map[string]string `json:"fingerprints,omitempty"`

	// An array of 'fix' objects, each of which represents a proposed fix to the problem indicated by the result.
	Fixes []*Fix `json:"fixes,omitempty"`

	// An array of one or more unique 'graphTraversal' objects.
	GraphTraversals []*GraphTraversal `json:"graphTraversals,omitempty"`

	// An array of zero or more unique graph objects associated with the result.
	Graphs []*Graph `json:"graphs,omitempty"`

	// A stable, unique identifer for the result in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// An absolute URI at which the result can be viewed.
	HostedViewerUri string `json:"hostedViewerUri,omitempty"`

	// A value that categorizes results by evaluation state.
	Kind interface{} `json:"kind,omitempty"`

	// A value specifying the severity level of the result.
	Level interface{} `json:"level,omitempty"`

	// The set of locations where the result was detected. Specify only one location unless the problem indicated by the result can only be corrected by making a change at every specified location.
	Locations []*Location `json:"locations,omitempty"`

	// A message that describes the result. The first sentence of the message only will be displayed when visible space is limited.
	Message *Message `json:"message"`

	// A positive integer specifying the number of times this logically unique result was observed in this run.
	OccurrenceCount int `json:"occurrenceCount,omitempty"`

	// A set of strings that contribute to the stable, unique identity of the result.
	PartialFingerprints map[string]string `json:"partialFingerprints,omitempty"`

	// Key/value pairs that provide additional information about the result.
	Properties *PropertyBag `json:"properties,omitempty"`

	// Information about how and when the result was detected.
	Provenance *ResultProvenance `json:"provenance,omitempty"`

	// A number representing the priority or importance of the result.
	Rank float64 `json:"rank,omitempty"`

	// A set of locations relevant to this result.
	RelatedLocations []*Location `json:"relatedLocations,omitempty"`

	// A reference used to locate the rule descriptor relevant to this result.
	Rule *ReportingDescriptorReference `json:"rule,omitempty"`

	// The stable, unique identifier of the rule, if any, to which this result is relevant.
	RuleId string `json:"ruleId,omitempty"`

	// The index within the tool component rules array of the rule object associated with this result.
	RuleIndex int `json:"ruleIndex,omitempty"`

	// An array of 'stack' objects relevant to the result.
	Stacks []*Stack `json:"stacks,omitempty"`

	// A set of suppressions relevant to this result.
	Suppressions []*Suppression `json:"suppressions,omitempty"`

	// An array of references to taxonomy reporting descriptors that are applicable to the result.
	Taxa []*ReportingDescriptorReference `json:"taxa,omitempty"`

	// A web request associated with this result.
	WebRequest *WebRequest `json:"webRequest,omitempty"`

	// A web response associated with this result.
	WebResponse *WebResponse `json:"webResponse,omitempty"`

	// The URIs of the work items associated with this result.
	WorkItemUris []string `json:"workItemUris,omitempty"`
}

// ResultProvenance Contains information about how and when a result was detected.
type ResultProvenance struct {

	// An array of physicalLocation objects which specify the portions of an analysis tool's output that a converter transformed into the result.
	ConversionSources []*PhysicalLocation `json:"conversionSources,omitempty"`

	// A GUID-valued string equal to the automationDetails.guid property of the run in which the result was first detected.
	FirstDetectionRunGuid string `json:"firstDetectionRunGuid,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which the result was first detected. See "Date/time properties" in the SARIF spec for the required format.
	FirstDetectionTimeUtc string `json:"firstDetectionTimeUtc,omitempty"`

	// The index within the run.invocations array of the invocation object which describes the tool invocation that detected the result.
	InvocationIndex int `json:"invocationIndex,omitempty"`

	// A GUID-valued string equal to the automationDetails.guid property of the run in which the result was most recently detected.
	LastDetectionRunGuid string `json:"lastDetectionRunGuid,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which the result was most recently detected. See "Date/time properties" in the SARIF spec for the required format.
	LastDetectionTimeUtc string `json:"lastDetectionTimeUtc,omitempty"`

	// Key/value pairs that provide additional information about the result.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// Run Describes a single run of an analysis tool, and contains the reported output of that run.
type Run struct {

	// Addresses associated with this run instance, if any.
	Addresses []*Address `json:"addresses,omitempty"`

	// An array of artifact objects relevant to the run.
	Artifacts []*Artifact `json:"artifacts,omitempty"`

	// Automation details that describe this run.
	AutomationDetails *RunAutomationDetails `json:"automationDetails,omitempty"`

	// The 'guid' property of a previous SARIF 'run' that comprises the baseline that was used to compute result 'baselineState' properties for the run.
	BaselineGuid string `json:"baselineGuid,omitempty"`

	// Specifies the unit in which the tool measures columns.
	ColumnKind interface{} `json:"columnKind,omitempty"`

	// A conversion object that describes how a converter transformed an analysis tool's native reporting format into the SARIF format.
	Conversion *Conversion `json:"conversion,omitempty"`

	// Specifies the default encoding for any artifact object that refers to a text file.
	DefaultEncoding string `json:"defaultEncoding,omitempty"`

	// Specifies the default source language for any artifact object that refers to a text file that contains source code.
	DefaultSourceLanguage string `json:"defaultSourceLanguage,omitempty"`

	// References to external property files that should be inlined with the content of a root log file.
	ExternalPropertyFileReferences *ExternalPropertyFileReferences `json:"externalPropertyFileReferences,omitempty"`

	// An array of zero or more unique graph objects associated with the run.
	Graphs []*Graph `json:"graphs,omitempty"`

	// Describes the invocation of the analysis tool.
	Invocations []*Invocation `json:"invocations,omitempty"`

	// The language of the messages emitted into the log file during this run (expressed as an ISO 639-1 two-letter lowercase culture code) and an optional region (expressed as an ISO 3166-1 two-letter uppercase subculture code associated with a country or region). The casing is recommended but not required (in order for this data to conform to RFC5646).
	Language string `json:"language,omitempty"`

	// An array of logical locations such as namespaces, types or functions.
	LogicalLocations []*LogicalLocation `json:"logicalLocations,omitempty"`

	// An ordered list of character sequences that were treated as line breaks when computing region information for the run.
	NewlineSequences []string `json:"newlineSequences,omitempty"`

	// The artifact location specified by each uriBaseId symbol on the machine where the tool originally ran.
	OriginalUriBaseIds map[string]*ArtifactLocation `json:"originalUriBaseIds,omitempty"`

	// Contains configurations that may potentially override both reportingDescriptor.defaultConfiguration (the tool's default severities) and invocation.configurationOverrides (severities established at run-time from the command line).
	Policies []*ToolComponent `json:"policies,omitempty"`

	// Key/value pairs that provide additional information about the run.
	Properties *PropertyBag `json:"properties,omitempty"`

	// An array of strings used to replace sensitive information in a redaction-aware property.
	RedactionTokens []string `json:"redactionTokens,omitempty"`

	// The set of results contained in an SARIF log. The results array can be omitted when a run is solely exporting rules metadata. It must be present (but may be empty) if a log file represents an actual scan.
	Results []*Result `json:"results,omitempty"`

	// Automation details that describe the aggregate of runs to which this run belongs.
	RunAggregates []*RunAutomationDetails `json:"runAggregates,omitempty"`

	// A specialLocations object that defines locations of special significance to SARIF consumers.
	SpecialLocations *SpecialLocations `json:"specialLocations,omitempty"`

	// An array of toolComponent objects relevant to a taxonomy in which results are categorized.
	Taxonomies []*ToolComponent `json:"taxonomies,omitempty"`

	// An array of threadFlowLocation objects cached at run level.
	ThreadFlowLocations []*ThreadFlowLocation `json:"threadFlowLocations,omitempty"`

	// Information about the tool or tool pipeline that generated the results in this run. A run can only contain results produced by a single tool or tool pipeline. A run can aggregate results from multiple log files, as long as context around the tool run (tool command-line arguments and the like) is identical for all aggregated files.
	Tool *Tool `json:"tool"`

	// The set of available translations of the localized data provided by the tool.
	Translations []*ToolComponent `json:"translations,omitempty"`

	// Specifies the revision in version control of the artifacts that were scanned.
	VersionControlProvenance []*VersionControlDetails `json:"versionControlProvenance,omitempty"`

	// An array of request objects cached at run level.
	WebRequests []*WebRequest `json:"webRequests,omitempty"`

	// An array of response objects cached at run level.
	WebResponses []*WebResponse `json:"webResponses,omitempty"`
}

// RunAutomationDetails Information that describes a run's identity and role within an engineering system process.
type RunAutomationDetails struct {

	// A stable, unique identifier for the equivalence class of runs to which this object's containing run object belongs in the form of a GUID.
	CorrelationGuid string `json:"correlationGuid,omitempty"`

	// A description of the identity and role played within the engineering system by this object's containing run object.
	Description *Message `json:"description,omitempty"`

	// A stable, unique identifer for this object's containing run object in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// A hierarchical string that uniquely identifies this object's containing run object.
	Id string `json:"id,omitempty"`

	// Key/value pairs that provide additional information about the run automation details.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// SpecialLocations Defines locations of special significance to SARIF consumers.
type SpecialLocations struct {

	// Provides a suggestion to SARIF consumers to display file paths relative to the specified location.
	DisplayBase *ArtifactLocation `json:"displayBase,omitempty"`

	// Key/value pairs that provide additional information about the special locations.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// Stack A call stack that is relevant to a result.
type Stack struct {

	// An array of stack frames that represents a sequence of calls, rendered in reverse chronological order, that comprise the call stack.
	Frames []*StackFrame `json:"frames"`

	// A message relevant to this call stack.
	Message *Message `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the stack.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// StackFrame A function call within a stack trace.
type StackFrame struct {

	// The location to which this stack frame refers.
	Location *Location `json:"location,omitempty"`

	// The name of the module that contains the code of this stack frame.
	Module string `json:"module,omitempty"`

	// The parameters of the call that is executing.
	Parameters []string `json:"parameters,omitempty"`

	// Key/value pairs that provide additional information about the stack frame.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The thread identifier of the stack frame.
	ThreadId int `json:"threadId,omitempty"`
}

// Suppression A suppression that is relevant to a result.
type Suppression struct {

	// A stable, unique identifer for the supression in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// A string representing the justification for the suppression.
	Justification string `json:"justification,omitempty"`

	// A string that indicates where the suppression is persisted.
	Kind string `json:"kind"`

	// Identifies the location associated with the suppression.
	Location *Location `json:"location,omitempty"`

	// Key/value pairs that provide additional information about the suppression.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A string that indicates the review status of the suppression.
	Status interface{} `json:"status,omitempty"`
}

// ThreadFlow Describes a sequence of code locations that specify a path through a single thread of execution such as an operating system or fiber.
type ThreadFlow struct {

	// An string that uniquely identifies the threadFlow within the codeFlow in which it occurs.
	Id string `json:"id,omitempty"`

	// Values of relevant expressions at the start of the thread flow that remain constant.
	ImmutableState map[string]*MultiformatMessageString `json:"immutableState,omitempty"`

	// Values of relevant expressions at the start of the thread flow that may change during thread flow execution.
	InitialState map[string]*MultiformatMessageString `json:"initialState,omitempty"`

	// A temporally ordered array of 'threadFlowLocation' objects, each of which describes a location visited by the tool while producing the result.
	Locations []*ThreadFlowLocation `json:"locations"`

	// A message relevant to the thread flow.
	Message *Message `json:"message,omitempty"`

	// Key/value pairs that provide additional information about the thread flow.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// ThreadFlowLocation A location visited by an analysis tool while simulating or monitoring the execution of a program.
type ThreadFlowLocation struct {

	// An integer representing the temporal order in which execution reached this location.
	ExecutionOrder int `json:"executionOrder,omitempty"`

	// The Coordinated Universal Time (UTC) date and time at which this location was executed.
	ExecutionTimeUtc string `json:"executionTimeUtc,omitempty"`

	// Specifies the importance of this location in understanding the code flow in which it occurs. The order from most to least important is "essential", "important", "unimportant". Default: "important".
	Importance interface{} `json:"importance,omitempty"`

	// The index within the run threadFlowLocations array.
	Index int `json:"index,omitempty"`

	// A set of distinct strings that categorize the thread flow location. Well-known kinds include 'acquire', 'release', 'enter', 'exit', 'call', 'return', 'branch', 'implicit', 'false', 'true', 'caution', 'danger', 'unknown', 'unreachable', 'taint', 'function', 'handler', 'lock', 'memory', 'resource', 'scope' and 'value'.
	Kinds []string `json:"kinds,omitempty"`

	// The code location.
	Location *Location `json:"location,omitempty"`

	// The name of the module that contains the code that is executing.
	Module string `json:"module,omitempty"`

	// An integer representing a containment hierarchy within the thread flow.
	NestingLevel int `json:"nestingLevel,omitempty"`

	// Key/value pairs that provide additional information about the threadflow location.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The call stack leading to this location.
	Stack *Stack `json:"stack,omitempty"`

	// A dictionary, each of whose keys specifies a variable or expression, the associated value of which represents the variable or expression value. For an annotation of kind 'continuation', for example, this dictionary might hold the current assumed values of a set of global variables.
	State map[string]*MultiformatMessageString `json:"state,omitempty"`

	// An array of references to rule or taxonomy reporting descriptors that are applicable to the thread flow location.
	Taxa []*ReportingDescriptorReference `json:"taxa,omitempty"`

	// A web request associated with this thread flow location.
	WebRequest *WebRequest `json:"webRequest,omitempty"`

	// A web response associated with this thread flow location.
	WebResponse *WebResponse `json:"webResponse,omitempty"`
}

// Tool The analysis tool that was run.
type Tool struct {

	// The analysis tool that was run.
	Driver *ToolComponent `json:"driver"`

	// Tool extensions that contributed to or reconfigured the analysis tool that was run.
	Extensions []*ToolComponent `json:"extensions,omitempty"`

	// Key/value pairs that provide additional information about the tool.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// ToolComponent A component, such as a plug-in or the driver, of the analysis tool that was run.
type ToolComponent struct {

	// The component which is strongly associated with this component. For a translation, this refers to the component which has been translated. For an extension, this is the driver that provides the extension's plugin model.
	AssociatedComponent *ToolComponentReference `json:"associatedComponent,omitempty"`

	// The kinds of data contained in this object.
	Contents []interface{} `json:"contents,omitempty"`

	// The binary version of the tool component's primary executable file expressed as four non-negative integers separated by a period (for operating systems that express file versions in this way).
	DottedQuadFileVersion string `json:"dottedQuadFileVersion,omitempty"`

	// The absolute URI from which the tool component can be downloaded.
	DownloadUri string `json:"downloadUri,omitempty"`

	// A comprehensive description of the tool component.
	FullDescription *MultiformatMessageString `json:"fullDescription,omitempty"`

	// The name of the tool component along with its version and any other useful identifying information, such as its locale.
	FullName string `json:"fullName,omitempty"`

	// A dictionary, each of whose keys is a resource identifier and each of whose values is a multiformatMessageString object, which holds message strings in plain text and (optionally) Markdown format. The strings can include placeholders, which can be used to construct a message in combination with an arbitrary number of additional string arguments.
	GlobalMessageStrings map[string]*MultiformatMessageString `json:"globalMessageStrings,omitempty"`

	// A unique identifer for the tool component in the form of a GUID.
	Guid string `json:"guid,omitempty"`

	// The absolute URI at which information about this version of the tool component can be found.
	InformationUri string `json:"informationUri,omitempty"`

	// Specifies whether this object contains a complete definition of the localizable and/or non-localizable data for this component, as opposed to including only data that is relevant to the results persisted to this log file.
	IsComprehensive bool `json:"isComprehensive,omitempty"`

	// The language of the messages emitted into the log file during this run (expressed as an ISO 639-1 two-letter lowercase language code) and an optional region (expressed as an ISO 3166-1 two-letter uppercase subculture code associated with a country or region). The casing is recommended but not required (in order for this data to conform to RFC5646).
	Language string `json:"language,omitempty"`

	// The semantic version of the localized strings defined in this component; maintained by components that provide translations.
	LocalizedDataSemanticVersion string `json:"localizedDataSemanticVersion,omitempty"`

	// An array of the artifactLocation objects associated with the tool component.
	Locations []*ArtifactLocation `json:"locations,omitempty"`

	// The minimum value of localizedDataSemanticVersion required in translations consumed by this component; used by components that consume translations.
	MinimumRequiredLocalizedDataSemanticVersion string `json:"minimumRequiredLocalizedDataSemanticVersion,omitempty"`

	// The name of the tool component.
	Name string `json:"name"`

	// An array of reportingDescriptor objects relevant to the notifications related to the configuration and runtime execution of the tool component.
	Notifications []*ReportingDescriptor `json:"notifications,omitempty"`

	// The organization or company that produced the tool component.
	Organization string `json:"organization,omitempty"`

	// A product suite to which the tool component belongs.
	Product string `json:"product,omitempty"`

	// A localizable string containing the name of the suite of products to which the tool component belongs.
	ProductSuite string `json:"productSuite,omitempty"`

	// Key/value pairs that provide additional information about the tool component.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A string specifying the UTC date (and optionally, the time) of the component's release.
	ReleaseDateUtc string `json:"releaseDateUtc,omitempty"`

	// An array of reportingDescriptor objects relevant to the analysis performed by the tool component.
	Rules []*ReportingDescriptor `json:"rules,omitempty"`

	// The tool component version in the format specified by Semantic Versioning 2.0.
	SemanticVersion string `json:"semanticVersion,omitempty"`

	// A brief description of the tool component.
	ShortDescription *MultiformatMessageString `json:"shortDescription,omitempty"`

	// An array of toolComponentReference objects to declare the taxonomies supported by the tool component.
	SupportedTaxonomies []*ToolComponentReference `json:"supportedTaxonomies,omitempty"`

	// An array of reportingDescriptor objects relevant to the definitions of both standalone and tool-defined taxonomies.
	Taxa []*ReportingDescriptor `json:"taxa,omitempty"`

	// Translation metadata, required for a translation, not populated by other component types.
	TranslationMetadata *TranslationMetadata `json:"translationMetadata,omitempty"`

	// The tool component version, in whatever format the component natively provides.
	Version string `json:"version,omitempty"`
}

// ToolComponentReference Identifies a particular toolComponent object, either the driver or an extension.
type ToolComponentReference struct {

	// The 'guid' property of the referenced toolComponent.
	Guid string `json:"guid,omitempty"`

	// An index into the referenced toolComponent in tool.extensions.
	Index int `json:"index,omitempty"`

	// The 'name' property of the referenced toolComponent.
	Name string `json:"name,omitempty"`

	// Key/value pairs that provide additional information about the toolComponentReference.
	Properties *PropertyBag `json:"properties,omitempty"`
}

// TranslationMetadata Provides additional metadata related to translation.
type TranslationMetadata struct {

	// The absolute URI from which the translation metadata can be downloaded.
	DownloadUri string `json:"downloadUri,omitempty"`

	// A comprehensive description of the translation metadata.
	FullDescription *MultiformatMessageString `json:"fullDescription,omitempty"`

	// The full name associated with the translation metadata.
	FullName string `json:"fullName,omitempty"`

	// The absolute URI from which information related to the translation metadata can be downloaded.
	InformationUri string `json:"informationUri,omitempty"`

	// The name associated with the translation metadata.
	Name string `json:"name"`

	// Key/value pairs that provide additional information about the translation metadata.
	Properties *PropertyBag `json:"properties,omitempty"`

	// A brief description of the translation metadata.
	ShortDescription *MultiformatMessageString `json:"shortDescription,omitempty"`
}

// VersionControlDetails Specifies the information necessary to retrieve a desired revision from a version control system.
type VersionControlDetails struct {

	// A Coordinated Universal Time (UTC) date and time that can be used to synchronize an enlistment to the state of the repository at that time.
	AsOfTimeUtc string `json:"asOfTimeUtc,omitempty"`

	// The name of a branch containing the revision.
	Branch string `json:"branch,omitempty"`

	// The location in the local file system to which the root of the repository was mapped at the time of the analysis.
	MappedTo *ArtifactLocation `json:"mappedTo,omitempty"`

	// Key/value pairs that provide additional information about the version control details.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The absolute URI of the repository.
	RepositoryUri string `json:"repositoryUri"`

	// A string that uniquely and permanently identifies the revision within the repository.
	RevisionId string `json:"revisionId,omitempty"`

	// A tag that has been applied to the revision.
	RevisionTag string `json:"revisionTag,omitempty"`
}

// WebRequest Describes an HTTP request.
type WebRequest struct {

	// The body of the request.
	Body *ArtifactContent `json:"body,omitempty"`

	// The request headers.
	Headers map[string]string `json:"headers,omitempty"`

	// The index within the run.webRequests array of the request object associated with this result.
	Index int `json:"index,omitempty"`

	// The HTTP method. Well-known values are 'GET', 'PUT', 'POST', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'TRACE', 'CONNECT'.
	Method string `json:"method,omitempty"`

	// The request parameters.
	Parameters map[string]string `json:"parameters,omitempty"`

	// Key/value pairs that provide additional information about the request.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The request protocol. Example: 'http'.
	Protocol string `json:"protocol,omitempty"`

	// The target of the request.
	Target string `json:"target,omitempty"`

	// The request version. Example: '1.1'.
	Version string `json:"version,omitempty"`
}

// WebResponse Describes the response to an HTTP request.
type WebResponse struct {

	// The body of the response.
	Body *ArtifactContent `json:"body,omitempty"`

	// The response headers.
	Headers map[string]string `json:"headers,omitempty"`

	// The index within the run.webResponses array of the response object associated with this result.
	Index int `json:"index,omitempty"`

	// Specifies whether a response was received from the server.
	NoResponseReceived bool `json:"noResponseReceived,omitempty"`

	// Key/value pairs that provide additional information about the response.
	Properties *PropertyBag `json:"properties,omitempty"`

	// The response protocol. Example: 'http'.
	Protocol string `json:"protocol,omitempty"`

	// The response reason. Example: 'Not found'.
	ReasonPhrase string `json:"reasonPhrase,omitempty"`

	// The response status code. Example: 451.
	StatusCode int `json:"statusCode,omitempty"`

	// The response version. Example: '1.1'.
	Version string `json:"version,omitempty"`
}
