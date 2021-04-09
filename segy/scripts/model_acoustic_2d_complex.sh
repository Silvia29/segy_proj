#!/bin/bash
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export MKL_DYNAMIC=true

datadir=${PWD}/data_cmp
modeldir=${PWD}/model_cmp
fdelmodc=${HOME}/bin/fdelmodc
filwav=$modeldir/wav.su

# Extend of model and modelling
z=0
cp0=1800
ro0=2100
cs0=900
vpvs=2000
xmax=10000
zmax=4000
zmax_ext=4200
tmax=4

time_delay=0.1
makewave file_out=$filwav\
	dt=.0004 \
	nt=1024 \
	w=g1  \
	t0=$time_delay \
	fp=30 \ 
	verbose=1 

j=0
while [ $j -lt 1 ] ; do

N=100
i=0
z=0
while [ $i -le $N ] ; do
    vm=0
    vm2=`expr \( 1000 \* $vm \) \/ 2`
    if [ $i -lt 3 -o $i -gt 5 ] ; then
        vp=$(( $vm + (RANDOM%250-125)))
    elif [ $i -eq 3 ] ; then
        vp=`expr $vp + $vm`
    elif [ $i -eq 4 ] ; then
        vp=$(($vp + (RANDOM%250-125)))
    elif [ $i -eq 5 ] ; then
        vp=`expr $vp - $vm2`
    fi
#    vp=`expr \( $vp + $i \* 25 \)`

    ro=$((2600+(RANDOM%1000-500)))
    vs=`expr \( 1000 \* $vp \) \/ $vpvs`
    if [ $z -lt 800 ] ; then
       z=`expr $z + 100 + $(((RANDOM%20-10)))` 
    elif [ $z -ge 800 ] ; then
       z=`expr $z + 250 + $(((RANDOM%100-50)))` 
    fi
    if [ $z -gt $zmax_ext ] ; then
        break
    fi

    # Create array of text strings for makemod
    random_layers[i+1]="cp=$vp cs=$vs ro=$ro grad=0 poly=0 intt=def x=0,$xmax z=$z,$z gradcp=0"
    echo $i ${random_layers[$i+1]}

    i=`expr $i + 1`

done
    
echo "**************************************************************"
echo "*                                                            *"
echo "*                                                            *"
echo "*               Model ${j}, number of layers = $N            *"
echo "*                                                            *"
echo "*                                                            *"
echo "**************************************************************"
i=0
dz=5
# Velcoity perturbations
makemod \
   file_base=tmp1.su \
   above=1 \
   dx=$dz \
   dz=$dz \
   cp0=$cp0 \
   cs0=$cs0 \
   ro0=$ro0 \
   sizex=$xmax \
   sizez=$zmax_ext \
   orig=0,0 \
   ${random_layers[*]} \
   verbose=0

# Velocity trend
gradcp=`bc -l <<< "scale=2 ; 1 + 2.5 * $((RANDOM % 10000 ))/10000"`
gradcs=`bc -l <<< "scale=2 ; $gradcp / 2"`
makemod \
   file_base=tmp2.su \
   above=0 \
   dx=$dz \
   dz=$dz \
   cp0=$cp0 \
   cs0=$cs0 \
   ro0=$ro0 \
   cp=$cp0 \
   cs=$cs0 \
   ro=$ro0 \
   sizex=$xmax \
   sizez=$zmax_ext \
   orig=0,0 \
   intt=def \
   x=0,$xmax \
   z=0,0 \
   gradcp=$gradcp \
   gradcs=$gradcs \
   verbose=0

suwind < tmp1_ro.su tmax=$zmax                  | sushw key=fldr a=$j > $modeldir/model_${j}_velan_ro_vintz_1d.su
susum tmp1_cp.su tmp2_cp.su | suwind tmax=$zmax | sushw key=fldr a=$j > $modeldir/model_${j}_velan_cp_vintz_1d.su
susum tmp1_cs.su tmp2_cs.su | suwind tmax=$zmax | sushw key=fldr a=$j > $modeldir/model_${j}_velan_cs_vintz_1d.su

#suwind < tmp1_cp.su key=tracl min=1 max=1 | sushw key=fldr a=1 | scale b=-$cp0 > tmp11.su
suwind < $modeldir/model_${j}_velan_cp_vintz_1d.su key=tracl min=1 max=1 > tmp_cp.su
suwind < $modeldir/model_${j}_velan_cs_vintz_1d.su key=tracl min=1 max=1 > tmp_cs.su
suwind < $modeldir/model_${j}_velan_ro_vintz_1d.su key=tracl min=1 max=1 > tmp_ro.su

#suxgraph < tmp_cp.su x2beg=0 x2end=6000 title="Vp: Vint vs depth" &
#suxgraph < tmp_cs.su x2beg=0 x2end=6000 title="Vs: Vint vs depth" &
#suxgraph < tmp_ro.su x2beg=0 x2end=6000 title="Rho vs depth" &

#exit 0
echo "**************************************************************"
echo "           model psct  & vz   "
echo "**************************************************************"
#src_type=1   ... 1=P 2=Txz 3=Tzz 4=Txx ...
#ischeme=3    ... 1=acoustic, 2=visco-acoustic 3=elastic, 4=visco-elastic
#src_orient=1 ... orientation of the source
#                 - 1=monopole
#                 - 2=dipole +/- vertical oriented
#                 - 3=dipole - + horizontal oriented
# For the 4 boundaries the options are:  1=free 2=pml 3=rigid 4=taper
#top=1 ............. type of boundary on top edge of model
#left=2 ............ type of boundary on left edge of model
#right=2 ........... type of boundary on right edge of model
#bottom=2 .......... type of boundary on bottom edge of model

xrcv1=1000
xrcv2=7000
xsrc=4000
dxrcv=5
zsrc=10
zrcv=10

$fdelmodc \
	file_cp=$modeldir/model_${j}_velan_cp_vintz_1d.su \
	file_cs=$modeldir/model_${j}_velan_cs_vintz_1d.su \
	file_den=$modeldir/model_${j}_velan_ro_vintz_1d.su \
	file_src=$filwav \
	file_rcv=$datadir/rcv_model_${j}_velan.su \
	\
        ischeme=1 \
	tmod=$tmax \
	npml=35 \
        top=1 \
        left=2 \
        right=2 \
        bottom=2 \
        verbose=2 \
	\
	src_type=1 \
	src_orient=1 \
	src_injectionrate=0 \
        xsrc=$xsrc \
	zsrc=$zsrc \
        nshot=1 \
        fmax=40 \
	\
        xrcv1=$xrcv1 \
	xrcv2=$xrcv2 \
	dxrcv=$dxrcv \
	zrcv1=$zrcv \
	zrcv2=$zrcv \
	dzrcv=0 \
	dtrcv=.004 \
        rec_type_p=1 \
        rec_type_vz=1 \
        rec_int_vz=2 \
	rec_delay=$time_delay
	
   suwind < $datadir/rcv_model_${j}_velan_rvz.su key=tracl j=1 | suwind key=offset min=0 | \
      sushw key=fldr,d1,f2 a=$j,0.004,0 > $datadir/vz_model_${j}_velan.su
   #suwind < $datadir/vz_model_${j}_velan.su tmax=3.2 | suximage perc=95 title="Model velan, shot $i" &
 
   ns=`expr $zmax \/ $dz + 1`
   sustrip < tmp_cp.su > tmp_$$.bin
   velconv < tmp_$$.bin intype=vintz outtype=vrmst dt=0.004 dz=$dz | \
       suaddhead ns=$ns | sushw key=d1 a=0.004 | suwind tmax=3.2 dt=0.004 > $modeldir/model_${j}_velan_cp_vrmst_1d.su
   suxgraph < $modeldir/model_${j}_velan_cp_vrmst_1d.su x2beg=1500 x2end=4500 title="VRMS(t)" &

   fv=1500
   dv=10
   suvelan < $datadir/vz_model_${j}_velan.su nv=350 dv=$dv fv=$fv smute=1.5 dtratio=8 pwr=0.5 | suwind tmax=3.2 \
           > $datadir/vz_model_${j}_sembl.su
   suximage < $datadir/vz_model_${j}_sembl.su cmap=hsv8 f2=$fv d2=$dv d1=0.032 x2beg=1500 x2end=4500 &
   i=`expr $i + 1`
   
   \rm tmp_$$.bin
   
sleep 0.5
#\rm tmp*su

j=`expr $j + 1`

done

exit 0
