package ffjsoninception

import "strings"

// ConditionalWrite is a stack containing a number of pending writes
type ConditionalWrite struct {
	Queued []string
}

// Write will add a string to be written
func (w *ConditionalWrite) Write(s string) {
	w.Queued = append(w.Queued, s)
}

// DeleteLast will delete the last added write
func (w *ConditionalWrite) DeleteLast() {
	if len(w.Queued) == 0 {
		return
	}
	w.Queued = w.Queued[:len(w.Queued)-1]
}

// Last will return the last added write
func (w *ConditionalWrite) Last() string {
	if len(w.Queued) == 0 {
		return ""
	}
	return w.Queued[len(w.Queued)-1]
}

// Flush will return all queued writes, and return
// "" (empty string) in nothing has been queued
// "buf.WriteByte('" + byte + "')" + '\n' if one bute has been queued.
// "buf.WriteString(`" + string + "`)" + "\n" if more than one byte has been queued.
func (w *ConditionalWrite) Flush() string {
	combined := strings.Join(w.Queued, "")
	if len(combined) == 0 {
		return ""
	}

	w.Queued = nil
	if len(combined) == 1 {
		return "buf.WriteByte('" + combined + "')" + "\n"
	}
	return "buf.WriteString(`" + combined + "`)" + "\n"
}

func (w *ConditionalWrite) FlushTo(out string) string {
	out += w.Flush()
	return out
}

// WriteFlush will add a string and return the Flush result for the queue
func (w *ConditionalWrite) WriteFlush(s string) string {
	w.Write(s)
	return w.Flush()
}

// GetQueued will return the current queued content without flushing.
func (w *ConditionalWrite) GetQueued() string {
	t := w.Queued
	s := w.Flush()
	w.Queued = t
	return s
}
