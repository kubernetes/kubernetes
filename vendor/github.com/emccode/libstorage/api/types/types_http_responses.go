package types

// VolumeAttachResponse is the JSON response for attaching a volume to an
// instance.
type VolumeAttachResponse struct {
	Volume      *Volume `json:"volume"`
	AttachToken string  `json:"attachToken"`
}
