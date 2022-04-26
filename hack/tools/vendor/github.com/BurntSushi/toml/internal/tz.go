package internal

import "time"

// Timezones used for local datetime, date, and time TOML types.
//
// The exact way times and dates without a timezone should be interpreted is not
// well-defined in the TOML specification and left to the implementation. These
// defaults to current local timezone offset of the computer, but this can be
// changed by changing these variables before decoding.
//
// TODO:
// Ideally we'd like to offer people the ability to configure the used timezone
// by setting Decoder.Timezone and Encoder.Timezone; however, this is a bit
// tricky: the reason we use three different variables for this is to support
// round-tripping – without these specific TZ names we wouldn't know which
// format to use.
//
// There isn't a good way to encode this right now though, and passing this sort
// of information also ties in to various related issues such as string format
// encoding, encoding of comments, etc.
//
// So, for the time being, just put this in internal until we can write a good
// comprehensive API for doing all of this.
//
// The reason they're exported is because they're referred from in e.g.
// internal/tag.
//
// Note that this behaviour is valid according to the TOML spec as the exact
// behaviour is left up to implementations.
var (
	localOffset   = func() int { _, o := time.Now().Zone(); return o }()
	LocalDatetime = time.FixedZone("datetime-local", localOffset)
	LocalDate     = time.FixedZone("date-local", localOffset)
	LocalTime     = time.FixedZone("time-local", localOffset)
)
