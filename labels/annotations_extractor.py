import json


def extract_results(file_name: str):
    """ Extracts annotations per video and saves them into separate files"""
    with open(file_name) as file:
        data = json.load(file)

        # Get the file name
        for sample_data in data:
            video_name = sample_data["file_upload"][-13:][:9]
            results = sample_data["annotations"][0]["result"]

            # Loop over the objects
            out = {}
            for i, result in enumerate(results):
                annotations = result["value"]["sequence"]

                # Loop over dict of annotated frames
                for annot in annotations:
                    # If a dictionary key is missing
                    if str(annot["frame"] - 1) not in out:
                        out[str(annot["frame"] - 1)] = []

                    # Put the values into the dictionary
                    out[str(annot["frame"] - 1)].append(
                        {
                            "x": annot["x"],
                            "y": annot["y"],
                            "width": annot["width"],
                            "height:": annot["height"]
                        }
                    )

            # Save the output file
            with open(video_name + ".json", "w") as out_file:
                json.dump(out, out_file)


in_file = ""    # add the file path
extract_results(in_file)
