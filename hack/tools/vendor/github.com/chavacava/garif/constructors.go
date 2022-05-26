package garif

// NewAddress creates a valid Address
func NewAddress() *Address {
	return &Address{}
}

// NewArtifact creates a valid Artifact
func NewArtifact() *Artifact {
	return &Artifact{}
}

// NewArtifactChange creates a valid ArtifactChange
func NewArtifactChange(location *ArtifactLocation, replacements ...*Replacement) *ArtifactChange {
	return &ArtifactChange{
		ArtifactLocation: location,
		Replacements:     replacements,
	}
}

// NewArtifactContent creates a valid ArtifactContent
func NewArtifactContent() *ArtifactContent {
	return &ArtifactContent{}
}

// NewArtifactLocation creates a valid ArtifactLocation
func NewArtifactLocation() *ArtifactLocation {
	return &ArtifactLocation{}
}

// NewAttachment creates a valid Attachment
func NewAttachment(location *ArtifactLocation) *Attachment {
	return &Attachment{ArtifactLocation: location}
}

// NewCodeFlow creates a valid CodeFlow
func NewCodeFlow(threadFlows ...*ThreadFlow) *CodeFlow {
	return &CodeFlow{ThreadFlows: threadFlows}
}

// NewConfigurationOverride creates a valid ConfigurationOverride
func NewConfigurationOverride(configuration *ReportingConfiguration, descriptor *ReportingDescriptorReference) *ConfigurationOverride {
	return &ConfigurationOverride{
		Configuration: configuration,
		Descriptor:    descriptor,
	}
}

// NewConversion creates a valid Conversion
func NewConversion(tool *Tool) *Conversion {
	return &Conversion{Tool: tool}
}

// NewEdge creates a valid Edge
func NewEdge(id, sourceNodeId, targetNodeId string) *Edge {
	return &Edge{
		Id:           id,
		SourceNodeId: sourceNodeId,
		TargetNodeId: targetNodeId,
	}
}

// NewEdgeTraversal creates a valid EdgeTraversal
func NewEdgeTraversal(edgeId string) *EdgeTraversal {
	return &EdgeTraversal{
		EdgeId: edgeId,
	}
}

// NewException creates a valid Exception
func NewException() *Exception {
	return &Exception{}
}

// NewExternalProperties creates a valid ExternalProperties
func NewExternalProperties() *ExternalProperties {
	return &ExternalProperties{}
}

// NewExternalPropertyFileReference creates a valid ExternalPropertyFileReference
func NewExternalPropertyFileReference() *ExternalPropertyFileReference {
	return &ExternalPropertyFileReference{}
}

// NewExternalPropertyFileReferences creates a valid ExternalPropertyFileReferences
func NewExternalPropertyFileReferences() *ExternalPropertyFileReferences {
	return &ExternalPropertyFileReferences{}
}

// NewFix creates a valid Fix
func NewFix(artifactChanges ...*ArtifactChange) *Fix {
	return &Fix{
		ArtifactChanges: artifactChanges,
	}
}

// NewGraph creates a valid Graph
func NewGraph() *Graph {
	return &Graph{}
}

// NewGraphTraversal creates a valid GraphTraversal
func NewGraphTraversal() *GraphTraversal {
	return &GraphTraversal{}
}

// NewInvocation creates a valid Invocation
func NewInvocation(executionSuccessful bool) *Invocation {
	return &Invocation{
		ExecutionSuccessful: executionSuccessful,
	}
}

// NewLocation creates a valid Location
func NewLocation() *Location {
	return &Location{}
}

// NewLocationRelationship creates a valid LocationRelationship
func NewLocationRelationship(target int) *LocationRelationship {
	return &LocationRelationship{
		Target: target,
	}
}

type LogFileVersion string

const Version210 LogFileVersion = "2.1.0"

// NewLogFile creates a valid LogFile
func NewLogFile(runs []*Run, version LogFileVersion) *LogFile {
	return &LogFile{
		Runs:    runs,
		Version: version,
	}
}

// NewLogicalLocation creates a valid LogicalLocation
func NewLogicalLocation() *LogicalLocation {
	return &LogicalLocation{}
}

// NewMessage creates a valid Message
func NewMessage() *Message {
	return &Message{}
}

// NewMessageFromText creates a valid Message with the given text
func NewMessageFromText(text string) *Message {
	return &Message{
		Text: text,
	}
}

// NewMultiformatMessageString creates a valid MultiformatMessageString
func NewMultiformatMessageString(text string) *MultiformatMessageString {
	return &MultiformatMessageString{
		Text: text,
	}
}

// NewNode creates a valid Node
func NewNode(id string) *Node {
	return &Node{
		Id: id,
	}
}

// NewNotification creates a valid Notification
func NewNotification(message *Message) *Notification {
	return &Notification{
		Message: message,
	}
}

// NewPhysicalLocation creates a valid PhysicalLocation
func NewPhysicalLocation() *PhysicalLocation {
	return &PhysicalLocation{}
}

// NewPropertyBag creates a valid PropertyBag
func NewPropertyBag() *PropertyBag {
	return &PropertyBag{}
}

// NewRectangle creates a valid Rectangle
func NewRectangle() *Rectangle {
	return &Rectangle{}
}

// NewRegion creates a valid Region
func NewRegion() *Region {
	return &Region{}
}

// NewReplacement creates a valid Replacement
func NewReplacement(deletedRegion *Region) *Replacement {
	return &Replacement{
		DeletedRegion: deletedRegion,
	}
}

// NewReportingConfiguration creates a valid ReportingConfiguration
func NewReportingConfiguration() *ReportingConfiguration {
	return &ReportingConfiguration{}
}

// NewReportingDescriptor creates a valid ReportingDescriptor
func NewReportingDescriptor(id string) *ReportingDescriptor {
	return &ReportingDescriptor{
		Id: id,
	}
}

// NewRule is an alias for NewReportingDescriptor
func NewRule(id string) *ReportingDescriptor {
	return NewReportingDescriptor(id)
}

// NewReportingDescriptorReference creates a valid ReportingDescriptorReference
func NewReportingDescriptorReference() *ReportingDescriptorReference {
	return &ReportingDescriptorReference{}
}

// NewReportingDescriptorRelationship creates a valid ReportingDescriptorRelationship
func NewReportingDescriptorRelationship(target *ReportingDescriptorReference) *ReportingDescriptorRelationship {
	return &ReportingDescriptorRelationship{
		Target: target,
	}
}

// NewResult creates a valid Result
func NewResult(message *Message) *Result {
	return &Result{
		Message: message,
	}
}

// NewResultProvenance creates a valid ResultProvenance
func NewResultProvenance() *ResultProvenance {
	return &ResultProvenance{}
}

// NewRun creates a valid Run
func NewRun(tool *Tool) *Run {
	return &Run{
		Tool: tool,
	}
}

// NewRunAutomationDetails creates a valid RunAutomationDetails
func NewRunAutomationDetails() *RunAutomationDetails {
	return &RunAutomationDetails{}
}

// New creates a valid
func NewSpecialLocations() *SpecialLocations {
	return &SpecialLocations{}
}

// NewStack creates a valid Stack
func NewStack(frames ...*StackFrame) *Stack {
	return &Stack{
		Frames: frames,
	}
}

// NewStackFrame creates a valid StackFrame
func NewStackFrame() *StackFrame {
	return &StackFrame{}
}

// NewSuppression creates a valid Suppression
func NewSuppression(kind string) *Suppression {
	return &Suppression{
		Kind: kind,
	}
}

// NewThreadFlow creates a valid ThreadFlow
func NewThreadFlow(locations []*ThreadFlowLocation) *ThreadFlow {
	return &ThreadFlow{
		Locations: locations,
	}
}

// NewThreadFlowLocation creates a valid ThreadFlowLocation
func NewThreadFlowLocation() *ThreadFlowLocation {
	return &ThreadFlowLocation{}
}

// NewTool creates a valid Tool
func NewTool(driver *ToolComponent) *Tool {
	return &Tool{
		Driver: driver,
	}
}

// NewToolComponent creates a valid ToolComponent
func NewToolComponent(name string) *ToolComponent {
	return &ToolComponent{
		Name: name,
	}
}

// NewDriver is an alias for NewToolComponent
func NewDriver(name string) *ToolComponent {
	return NewToolComponent(name)
}

// NewToolComponentReference creates a valid ToolComponentReference
func NewToolComponentReference() *ToolComponentReference {
	return &ToolComponentReference{}
}

// NewTranslationMetadata creates a valid TranslationMetadata
func NewTranslationMetadata(name string) *TranslationMetadata {
	return &TranslationMetadata{
		Name: name,
	}
}

// NewVersionControlDetails creates a valid VersionControlDetails
func NewVersionControlDetails(repositoryUri string) *VersionControlDetails {
	return &VersionControlDetails{
		RepositoryUri: repositoryUri,
	}
}

// NewWebRequest creates a valid WebRequest
func NewWebRequest() *WebRequest {
	return &WebRequest{}
}

// NewWebResponse creates a valid WebResponse
func NewWebResponse() *WebResponse {
	return &WebResponse{}
}
