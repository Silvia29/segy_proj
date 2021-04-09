#!/bin/bash
csv_file=stkvel_analysis_files.csv
datadir=${PWD}/data_cmp
modeldir=${PWD}/model_cmp

# Convert to SEGY format
# Put Vp (cp), Vs (cs) and density (ro) files into a single su/sgy file

create_model=1
if [ $create_model -eq 1 ] ; then

	segyhdrs < $modeldir/model_0_velan_cp_vintz_1d.su dt=5000 \
	           bfile=binary_model hfile=header_model
	segyhdrs < $datadir/vz_model_0_velan.su \
	           bfile=binary_shot  hfile=header_shot
	suwind < $datadir/vz_model_0_sembl.su count=350 | sushw key=dt,d2,f2 a=4000,10,1500 | \
	    sushw key=offset a=1500 b=10 | segyhdrs bfile=binary_sembl  hfile=header_sembl
	      
	j=0
	while [ $j -lt 3 ] ; do
	   vint_cp="$modeldir/model_${j}_velan_cp_vintz_1d.su"
	   vint_cs="$modeldir/model_${j}_velan_cs_vintz_1d.su"
	   vint_ro="$modeldir/model_${j}_velan_ro_vintz_1d.su"
	   vrms_cp="$modeldir/model_${j}_velan_cp_vrmst_1d.su"
	   
	   velan="$datadir/vz_model_${j}_velan"
	   sembl="$datadir/vz_model_${j}_sembl"
	   mout="$modeldir/model_${j}_velan.sgy"
	   
#           if [ -s $vint_cp -a -s $vint_cs -a -s $vint_ro -a -s $vrms_cp -a -s ${velan}.su -a -s ${sembl}.su ] ; then
#	    if [ ! -e $mout -o ! -e ${velan}.sgy -o ! -e ${sembl}.sgy  ] ; then
              \rm -rf vrms_cp.su vint_cp.su vint_cs.su vint_ro.su model_velan.su
              echo "Processing experiment $j"
              sushw < $vrms_cp key=dt a=4000 | suwind key=tracl min=1 max=1 | sushw key=tracl a=1 > vrms_cp.su
              suwind < $vint_cp key=tracl min=1 max=1 | sushw key=tracl a=2 > vint_cp.su
              suwind < $vint_cs key=tracl min=1 max=1 | sushw key=tracl a=3 > vint_cs.su
              suwind < $vint_ro key=tracl min=1 max=1 | sushw key=tracl a=4 > vint_ro.su
              cat vrms_cp.su vint_cp.su vint_cs.su vint_ro.su >> model_velan.su

            # Output Vp(z), Vs(z), density(z) and Vrms(t) in single segy file
	      segywrite < model_velan.su bfile=binary_model hfile=header_model tape=$mout

            # Apply t0.5 gain to shot gather
	      suwind < ${velan}.su key=offset min=0 | sugain tpow=0.5 | \
               segywrite tape=${velan}.sgy bfile=binary_shot hfile=header_shot

	      suwind < ${sembl}.su count=350 | sushw key=dt a=4000 | sushw key=offset a=1500 b=10 | \
               segywrite tape=${sembl}.sgy bfile=binary_sembl hfile=header_sembl
#	    else
#	      echo "Files for $j exist"
#            fi
#	   fi
           j=`expr $j + 1`
	done
fi

exit 0
