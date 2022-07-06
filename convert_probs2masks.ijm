function convertProb2Mask(input_dir, input_file, output_dir){
    open(input_dir + input_file);
    setOption("BlackBackground", true);
    run("Convert to Mask", "method=Default background=Dark calculate black");
    run("Duplicate...", "use");
    selectWindow(input_file);
    close();
    selectWindow("cell");
    saveAs("Tiff", output_dir + input_file);
    close();
}

input_dir = getDirectory("Input Directory");
output_dir = getDirectory("Output Directory");
print("Input directory: "+ input_dir);
print("Output directory: "+ output_dir);

list = getFileList(input_dir);
file_cnt = 0;
for(i=0;i<list.length;i++)
{
	if (endsWith(list[i],".tif"))
	{
		convertProb2Mask(input_dir, list[i], output_dir);
		file_cnt++;
	}
}
print("Done! " + toString(file_cnt) + " prob file(s) was processed.");