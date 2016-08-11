// Package to work with VHD images
// See https://technet.microsoft.com/en-us/virtualization/bb676673.aspx
package vhd

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

const VHD_COOKIE = "636f6e6563746978"     // conectix
const VHD_DYN_COOKIE = "6378737061727365" // cxsparse
const VHD_CREATOR_APP = "676f2d766864"    // go-vhd
const VHD_CREATOR_HOST_OS = "5769326B"    // Win2k
const VHD_BLOCK_SIZE = 2 * 1024 * 1024    // 2MB
const VHD_HEADER_SIZE = 512
const SECTOR_SIZE = 512
const FOURK_SECTOR_SIZE = 4096
const VHD_EXTRA_HEADER_SIZE = 1024

// A VDH file
type VHD struct {
	Footer      VHDHeader
	ExtraHeader VHDExtraHeader
}

// VHD Header
type VHDHeader struct {
	Cookie             [8]byte
	Features           [4]byte
	FileFormatVersion  [4]byte
	DataOffset         [8]byte
	Timestamp          [4]byte
	CreatorApplication [4]byte
	CreatorVersion     [4]byte
	CreatorHostOS      [4]byte
	OriginalSize       [8]byte
	CurrentSize        [8]byte
	DiskGeometry       [4]byte
	DiskType           [4]byte
	Checksum           [4]byte
	UniqueId           [16]byte
	SavedState         [1]byte
	Reserved           [427]byte
}

// VHD extra header, for dynamic and differential disks
type VHDExtraHeader struct {
	Cookie              [8]byte
	DataOffset          [8]byte
	TableOffset         [8]byte
	HeaderVersion       [4]byte
	MaxTableEntries     [4]byte
	BlockSize           [4]byte
	Checksum            [4]byte
	ParentUUID          [16]byte
	ParentTimestamp     [4]byte
	Reserved            [4]byte
	ParentUnicodeName   [512]byte
	ParentLocatorEntry1 [24]byte
	ParentLocatorEntry2 [24]byte
	ParentLocatorEntry3 [24]byte
	ParentLocatorEntry4 [24]byte
	ParentLocatorEntry5 [24]byte
	ParentLocatorEntry6 [24]byte
	ParentLocatorEntry7 [24]byte
	ParentLocatorEntry8 [24]byte
	Reserved2           [256]byte
}

// Options for the CreateSparseVHD function
type VHDOptions struct {
	UUID      string
	Timestamp int64
}

/*
 *  VHDExtraHeader methods
 */

func (header *VHDExtraHeader) CookieString() string {
	return string(header.Cookie[:])
}

// Calculate and add the VHD dynamic/differential header checksum
func (h *VHDExtraHeader) addChecksum() {
	buffer := new(bytes.Buffer)
	binary.Write(buffer, binary.BigEndian, h)
	checksum := 0
	bb := buffer.Bytes()

	for counter := 0; counter < VHD_EXTRA_HEADER_SIZE; counter++ {
		checksum += int(bb[counter])
	}

	binary.BigEndian.PutUint32(h.Checksum[:], uint32(^checksum))
}

/*
 * VHDHeader methods
 */

func (h *VHDHeader) DiskTypeStr() (dt string) {
	switch h.DiskType[3] {
	case 0x00:
		dt = "None"
	case 0x01:
		dt = "Deprecated"
	case 0x02:
		dt = "Fixed"
	case 0x03:
		dt = "Dynamic"
	case 0x04:
		dt = "Differential"
	case 0x05:
		dt = "Reserved"
	case 0x06:
		dt = "Reserved"
	default:
		panic("Invalid disk type detected!")
	}

	return
}

// Return the timestamp of the header
func (h *VHDHeader) TimestampTime() time.Time {
	tstamp := binary.BigEndian.Uint32(h.Timestamp[:])
	return time.Unix(int64(946684800+tstamp), 0)
}

// Calculate and add the VHD header checksum
func (h *VHDHeader) addChecksum() {
	buffer := new(bytes.Buffer)
	binary.Write(buffer, binary.BigEndian, h)
	checksum := 0
	bb := buffer.Bytes()

	for counter := 0; counter < VHD_HEADER_SIZE; counter++ {
		checksum += int(bb[counter])
	}

	binary.BigEndian.PutUint32(h.Checksum[:], uint32(^checksum))
}

func CreateFixedHeader(size uint64, options *VHDOptions) VHDHeader {
	header := VHDHeader{}
	hexToField(VHD_COOKIE, header.Cookie[:])
	hexToField("00000002", header.Features[:])
	hexToField("00010000", header.FileFormatVersion[:])
	hexToField("ffffffffffffffff", header.DataOffset[:])

	// LOL Y2038
	if options.Timestamp != 0 {
		binary.BigEndian.PutUint32(header.Timestamp[:], uint32(options.Timestamp))
	} else {
		t := uint32(time.Now().Unix() - 946684800)
		binary.BigEndian.PutUint32(header.Timestamp[:], t)
	}

	hexToField(VHD_CREATOR_APP, header.CreatorApplication[:])
	hexToField(VHD_CREATOR_HOST_OS, header.CreatorHostOS[:])
	binary.BigEndian.PutUint64(header.OriginalSize[:], size)
	binary.BigEndian.PutUint64(header.CurrentSize[:], size)

	// total sectors = disk size / 512b sector size
	totalSectors := math.Floor(float64(size / 512))
	// [C, H, S]
	geometry := calculateCHS(uint64(totalSectors))
	binary.BigEndian.PutUint16(header.DiskGeometry[:2], uint16(geometry[0]))
	header.DiskGeometry[2] = uint8(geometry[1])
	header.DiskGeometry[3] = uint8(geometry[2])

	hexToField("00000002", header.DiskType[:]) // Fixed 0x00000002
	hexToField("00000000", header.Checksum[:])

	if options.UUID != "" {
		copy(header.UniqueId[:], uuidToBytes(options.UUID))
	} else {
		copy(header.UniqueId[:], uuidgenBytes())
	}

	header.addChecksum()
	return header
}

func RawToFixed(f *os.File, options *VHDOptions) {
	info, err := f.Stat()
	check(err)
	size := uint64(info.Size())
	header := CreateFixedHeader(size, options)
	binary.Write(f, binary.BigEndian, header)
}

func VHDCreateSparse(size uint64, name string, options VHDOptions) VHD {
	header := VHDHeader{}
	hexToField(VHD_COOKIE, header.Cookie[:])
	hexToField("00000002", header.Features[:])
	hexToField("00010000", header.FileFormatVersion[:])
	hexToField("0000000000000200", header.DataOffset[:])

	// LOL Y2038
	if options.Timestamp != 0 {
		binary.BigEndian.PutUint32(header.Timestamp[:], uint32(options.Timestamp))
	} else {
		t := uint32(time.Now().Unix() - 946684800)
		binary.BigEndian.PutUint32(header.Timestamp[:], t)
	}

	hexToField(VHD_CREATOR_APP, header.CreatorApplication[:])
	hexToField(VHD_CREATOR_HOST_OS, header.CreatorHostOS[:])
	binary.BigEndian.PutUint64(header.OriginalSize[:], size)
	binary.BigEndian.PutUint64(header.CurrentSize[:], size)

	// total sectors = disk size / 512b sector size
	totalSectors := math.Floor(float64(size / 512))
	// [C, H, S]
	geometry := calculateCHS(uint64(totalSectors))
	binary.BigEndian.PutUint16(header.DiskGeometry[:2], uint16(geometry[0]))
	header.DiskGeometry[2] = uint8(geometry[1])
	header.DiskGeometry[3] = uint8(geometry[2])

	hexToField("00000003", header.DiskType[:]) // Sparse 0x00000003
	hexToField("00000000", header.Checksum[:])

	if options.UUID != "" {
		copy(header.UniqueId[:], uuidToBytes(options.UUID))
	} else {
		copy(header.UniqueId[:], uuidgenBytes())
	}

	header.addChecksum()

	// Fill the sparse header
	header2 := VHDExtraHeader{}
	hexToField(VHD_DYN_COOKIE, header2.Cookie[:])
	hexToField("ffffffffffffffff", header2.DataOffset[:])
	// header size + sparse header size
	binary.BigEndian.PutUint64(header2.TableOffset[:], uint64(VHD_EXTRA_HEADER_SIZE+VHD_HEADER_SIZE))
	hexToField("00010000", header2.HeaderVersion[:])

	maxTableSize := uint32(size / (VHD_BLOCK_SIZE))
	binary.BigEndian.PutUint32(header2.MaxTableEntries[:], maxTableSize)

	binary.BigEndian.PutUint32(header2.BlockSize[:], VHD_BLOCK_SIZE)
	binary.BigEndian.PutUint32(header2.ParentTimestamp[:], uint32(0))
	header2.addChecksum()

	f, err := os.Create(name)
	check(err)
	defer f.Close()

	binary.Write(f, binary.BigEndian, header)
	binary.Write(f, binary.BigEndian, header2)

	/*
		Write BAT entries
		The BAT is always extended to a sector (4K) boundary
		1536 = 512 + 1024 (the VHD Header + VHD Sparse header size)
	*/
	for count := uint32(0); count < (FOURK_SECTOR_SIZE - 1536); count += 1 {
		f.Write([]byte{0xff})
	}

	/* Windows creates 8K VHDs by default */
	for i := 0; i < (FOURK_SECTOR_SIZE - VHD_HEADER_SIZE); i += 1 {
		f.Write([]byte{0x0})
	}

	binary.Write(f, binary.BigEndian, header)

	return VHD{
		Footer:      header,
		ExtraHeader: header2,
	}
}

/*
 * VHD
 */

func FromFile(f *os.File) (vhd VHD) {
	vhd = VHD{}
	vhd.Footer = readVHDFooter(f)
	vhd.ExtraHeader = readVHDExtraHeader(f)

	return vhd
}

func (vhd *VHD) PrintInfo() {
	fmt.Println("\nVHD footer")
	fmt.Println("==========")
	vhd.PrintFooter()

	if vhd.Footer.DiskType[3] == 0x3 || vhd.Footer.DiskType[3] == 0x04 {
		fmt.Println("\nVHD sparse/differential header")
		fmt.Println("===============================")
		vhd.PrintExtraHeader()
	}
}

func (vhd *VHD) PrintExtraHeader() {
	header := vhd.ExtraHeader

	fmtField("Cookie", fmt.Sprintf("%s (%s)",
		hexs(header.Cookie[:]), header.CookieString()))
	fmtField("Data offset", hexs(header.DataOffset[:]))
	fmtField("Table offset", hexs(header.TableOffset[:]))
	fmtField("Header version", hexs(header.HeaderVersion[:]))
	fmtField("Max table entries", hexs(header.MaxTableEntries[:]))
	fmtField("Block size", hexs(header.BlockSize[:]))
	fmtField("Checksum", hexs(header.Checksum[:]))
	fmtField("Parent UUID", uuid(header.ParentUUID[:]))

	// Seconds since January 1, 1970 12:00:00 AM in UTC/GMT.
	// 946684800 = January 1, 2000 12:00:00 AM in UTC/GMT.
	tstamp := binary.BigEndian.Uint32(header.ParentTimestamp[:])
	t := time.Unix(int64(946684800+tstamp), 0)
	fmtField("Parent timestamp", fmt.Sprintf("%s", t))

	fmtField("Reserved", hexs(header.Reserved[:]))
	parentName := utf16BytesToString(header.ParentUnicodeName[:],
		binary.BigEndian)
	fmtField("Parent Name", parentName)
	// Parent locator entries ignored since it's a dynamic disk
	sum := 0
	for _, b := range header.Reserved2 {
		sum += int(b)
	}
	fmtField("Reserved2", strconv.Itoa(sum))
}

func (vhd *VHD) PrintFooter() {
	header := vhd.Footer

	//fmtField("Cookie", string(header.Cookie[:]))
	fmtField("Cookie", fmt.Sprintf("%s (%s)",
		hexs(header.Cookie[:]), string(header.Cookie[:])))
	fmtField("Features", hexs(header.Features[:]))
	fmtField("File format version", hexs(header.FileFormatVersion[:]))

	dataOffset := binary.BigEndian.Uint64(header.DataOffset[:])
	fmtField("Data offset",
		fmt.Sprintf("%s (%d bytes)", hexs(header.DataOffset[:]), dataOffset))

	//// Seconds since January 1, 1970 12:00:00 AM in UTC/GMT.
	//// 946684800 = January 1, 2000 12:00:00 AM in UTC/GMT.
	t := time.Unix(int64(946684800+binary.BigEndian.Uint32(header.Timestamp[:])), 0)
	fmtField("Timestamp", fmt.Sprintf("%s", t))

	fmtField("Creator application", string(header.CreatorApplication[:]))
	fmtField("Creator version", hexs(header.CreatorVersion[:]))
	fmtField("Creator OS", string(header.CreatorHostOS[:]))

	originalSize := binary.BigEndian.Uint64(header.OriginalSize[:])
	fmtField("Original size",
		fmt.Sprintf("%s ( %d bytes )", hexs(header.OriginalSize[:]), originalSize))

	currentSize := binary.BigEndian.Uint64(header.OriginalSize[:])
	fmtField("Current size",
		fmt.Sprintf("%s ( %d bytes )", hexs(header.CurrentSize[:]), currentSize))

	cilinders := int64(binary.BigEndian.Uint16(header.DiskGeometry[:2]))
	heads := int64(header.DiskGeometry[2])
	sectors := int64(header.DiskGeometry[3])
	dsize := cilinders * heads * sectors * 512
	fmtField("Disk geometry",
		fmt.Sprintf("%s (c: %d, h: %d, s: %d) (%d bytes)",
			hexs(header.DiskGeometry[:]),
			cilinders,
			heads,
			sectors,
			dsize))

	fmtField("Disk type",
		fmt.Sprintf("%s (%s)", hexs(header.DiskType[:]), header.DiskTypeStr()))

	fmtField("Checksum", hexs(header.Checksum[:]))
	fmtField("UUID", uuid(header.UniqueId[:]))
	fmtField("Saved state", fmt.Sprintf("%d", header.SavedState[0]))
}

/*
	Utility functions
*/
func calculateCHS(ts uint64) []uint {
	var sectorsPerTrack,
		heads,
		cylinderTimesHeads,
		cylinders float64
	totalSectors := float64(ts)

	ret := make([]uint, 3)

	if totalSectors > 65535*16*255 {
		totalSectors = 65535 * 16 * 255
	}

	if totalSectors >= 65535*16*63 {
		sectorsPerTrack = 255
		heads = 16
		cylinderTimesHeads = math.Floor(totalSectors / sectorsPerTrack)
	} else {
		sectorsPerTrack = 17
		cylinderTimesHeads = math.Floor(totalSectors / sectorsPerTrack)
		heads = math.Floor((cylinderTimesHeads + 1023) / 1024)
		if heads < 4 {
			heads = 4
		}
		if (cylinderTimesHeads >= (heads * 1024)) || heads > 16 {
			sectorsPerTrack = 31
			heads = 16
			cylinderTimesHeads = math.Floor(totalSectors / sectorsPerTrack)
		}
		if cylinderTimesHeads >= (heads * 1024) {
			sectorsPerTrack = 63
			heads = 16
			cylinderTimesHeads = math.Floor(totalSectors / sectorsPerTrack)
		}
	}

	cylinders = cylinderTimesHeads / heads

	// This will floor the values
	ret[0] = uint(cylinders)
	ret[1] = uint(heads)
	ret[2] = uint(sectorsPerTrack)

	return ret
}

func hexToField(hexs string, field []byte) {
	h, err := hex.DecodeString(hexs)
	check(err)

	copy(field, h)
}

// Return the number of blocks in the disk, diskSize in bytes
func getMaxTableEntries(diskSize uint64) uint64 {
	return diskSize * (2 * 1024 * 1024) // block size is 2M
}

func readVHDExtraHeader(f *os.File) (header VHDExtraHeader) {
	buff := make([]byte, 1024)
	_, err := f.ReadAt(buff, 512)
	check(err)

	binary.Read(bytes.NewBuffer(buff[:]), binary.BigEndian, &header)

	return header
}

func readVHDFooter(f *os.File) (header VHDHeader) {
	info, err := f.Stat()
	check(err)

	buff := make([]byte, 512)
	_, err = f.ReadAt(buff, info.Size()-512)
	check(err)

	binary.Read(bytes.NewBuffer(buff[:]), binary.BigEndian, &header)

	return header
}

func readVHDHeader(f *os.File) (header VHDHeader) {
	buff := make([]byte, 512)
	_, err := f.ReadAt(buff, 0)
	check(err)

	binary.Read(bytes.NewBuffer(buff[:]), binary.BigEndian, &header)

	return header
}
