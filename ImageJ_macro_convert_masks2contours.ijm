// Before running this macro, copy cropped images to `output_dir`
function convertMask2Contour(input_dir, input_file, output_dir){
    open(input_dir + input_file);
    run("Analyze Particles...", "size=6-Infinity clear add");
    open(output_dir + input_file);
    run("From ROI Manager");
    saveAs("Tiff", output_dir + input_file);
    close();
    selectWindow(input_file);
    close();
    close("ROI Manager");
}

input_dir = getDirectory("Mask Directory");
output_dir = getDirectory("Contour Directory");
print("Mask directory: "+ input_dir);
print("Contour directory: "+ output_dir);

list = getFileList(input_dir);
file_cnt = 0;
for(i=0;i<list.length;i++)
{
	if (endsWith(list[i],".tif"))
	{		
		convertMask2Contour(input_dir, list[i], output_dir);
		file_cnt++;
	}
}
print("Done! " + toString(file_cnt) + " contour file(s) was processed.");