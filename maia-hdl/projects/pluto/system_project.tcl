
source ../../adi-hdl/scripts/adi_env.tcl
source $ad_hdl_dir/projects/scripts/adi_project_xilinx.tcl
source $ad_hdl_dir/projects/scripts/adi_board.tcl

adi_project_create pluto 0 {} "xc7z010clg225-1"

adi_project_files pluto [list \
  "system_top.v" \
  "system_constr.xdc" \
  "$ad_hdl_dir/library/common/ad_iobuf.v"]

# use improved implementation strategy for best timing results
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# Bake the git short hash (passed in via $GIT_HASH from build_firmware_full.sh)
# into the bitstream so the running gateware is identifiable: USERID shows in the
# .bit ASCII header (`strings system_top.bit | grep UserID`) and USR_ACCESS is
# readable at runtime. Defaults to 0xFFFFFFFF for ad-hoc/local builds with no env.
set git_hash "ffffffff"
if {[info exists ::env(GIT_HASH)] && [string trim $::env(GIT_HASH)] ne ""} {
    set git_hash [string trim $::env(GIT_HASH)]
}
puts "INFO: embedding fork commit 0x$git_hash into bitstream (USERID + USR_ACCESS)"
set_property -name {STEPS.WRITE_BITSTREAM.ARGS.MORE OPTIONS} \
    -value "-g USERID:0x$git_hash -g USR_ACCESS:0x$git_hash" \
    -objects [get_runs impl_1]

set_property is_enabled false [get_files  *system_sys_ps7_0.xdc]
adi_project_run pluto
source $ad_hdl_dir/library/axi_ad9361/axi_ad9361_delay.tcl
