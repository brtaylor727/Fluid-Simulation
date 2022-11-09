
#for j in {"uy","phi","divu","F","ux"}
for j in {"uy","phi","ux"}
do
	rm ./movie/*.png
#	for i in $(seq -f "%04g" 1 20)
#	do
   		#gnuplot -p -e "set terminal png size 1000,1000 enhanced font ',10';set view map;  set dgrid3d; set rmargin .6; splot './data/$j$i.dat' using 1:2:3 with pm3d;set pm3d interpolate 10,10" > ./movie/pic$i.png
#		python pythonplot.py "data/$j$i.dat"
	
#	done
#	python pplotall.py "data/$j"
	python pplotallparallel.py "data/$j" 8

	ffmpeg -y -r 4 -i movie/$j%04d.png -c:v libx264 -vf fps=20 $j.mp4


done

rm olddata/*
mv ./data/*.dat olddata



