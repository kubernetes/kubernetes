package dockerpre012

// DeepCopyInto is manually built to copy the (probably bugged) time.Time
func (in *ImagePre012) DeepCopyInto(out *ImagePre012) {
	*out = *in
	out.Created = in.Created
	in.ContainerConfig.DeepCopyInto(&out.ContainerConfig)
	if in.Config != nil {
		in, out := &in.Config, &out.Config
		if *in == nil {
			*out = nil
		} else {
			*out = new(Config)
			(*in).DeepCopyInto(*out)
		}
	}
	return
}
