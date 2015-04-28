//+build darwin

package tarheader

import (
	"archive/tar"
	"os"
	"syscall"
	"time"
)

func init() {
	populateHeaderStat = append(populateHeaderStat, populateHeaderCtime)
}

func populateHeaderCtime(h *tar.Header, fi os.FileInfo, _ map[uint64]string) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return
	}

	sec, nsec := st.Ctimespec.Unix()
	ctime := time.Unix(sec, nsec)
	h.ChangeTime = ctime
}
