#!/bin/bash

# For the moment k=2 is not supported since PBH would form in matter dominated universe
kn='4'                 # power of inflaton potential
path="./Results/k="$kn
mkdir -p $path          # directory for storing data for each yukawa coupling
yphi="1E00"            # yukawa coupling of  Φ --> f f  (y_eff Φff)
yukdir="yukawa=1E00"   

logMBHin="2"  # log10 of initial black hole mass
sigmaM="2"    # sigma = 0 for monochromatic BH "mono"
Mdist="ext"   # mass distribution ("ext": enxtended power-law, "mono": monochromatic)
beta="-10"    # Initial fraction of black hole energy density
tag="0"       # a tag 

kminus=$((kn-2))
kplus=$((kn+2))
omega=$(echo "scale=5; ($kminus / $kplus)" | bc -l)        
alpha=$(echo "scale=5; (((4*$omega+2))/(($omega+1)))" | bc -l)  

echo "#----------------------------------------------------#"
echo "       PBH  +  Φ  with potential V(Φ) = Φ^"$kn
echo " "
echo "In this version k=2 is not supported, since PBH would"
echo "         form in matter dominated universe.          "
echo "                    Chose k > 2                      "
echo " PBH evaporate to SM (no Dark matter). See the other "
echo "                    package                          "
echo "#----------------------------------------------------#"
echo " "


echo "omega, yphi, alpha = ", $omega, $yphi, $alpha

touch tempyuk.dat
echo $yphi	$kn	$logMBHin > tempyuk.dat
			
#python3 -W ignore example_DM_MassDist.py $logmDM $beta $logMBHin $sigmaM $alpha $Mdist $yphi
mkdir -p $path/phiff/$yuk/$MBHin/databeta=$beta/sigma_2
python3 -W ignore script_scan.py $tag $beta $logMBHin $sigmaM $alpha $Mdist $yphi 
mv $path/data_scan*  $path/phiff/$yuk/$MBHin/databeta=$beta/sigma_$sigmaM/
