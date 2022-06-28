#get wanted list

def getWantedLists(wanted_key, numu_stacked):
        numu_wanted_list = []
        import numpy
        sumlist = []
        zlist = []
        onelist = []
        twolist = []
        threelist = []
        fourlist = []
        fivelist = []
        sixlist = []
        sevenlist = []
        eightlist = []
        ninelist = []
        for i in numu_stacked:
            for j in i:
                #print(j)
                if (type(j) == numpy.ndarray):
                    #print("sum: ", sum(j))
                    sumlist.append(sum(j))
                    zlist.append(j[0])
                    onelist.append(j[1])
                    twolist.append(j[2])
                    threelist.append(j[3])
                    fourlist.append(j[4])
                    fivelist.append(j[5])
                    sixlist.append(j[6])
                    sevenlist.append(j[7])
                    eightlist.append(j[8])
                    ninelist.append(j[9])
        
        #print("")
        #print("ZERO DIFF")
        #print(zlist)
        zdiff = [abs(k-l) for k,l in zip(zlist[:-1], zlist[1:])]
        #print(zdiff)
        #print(len(zdiff))
        if (len(zlist) != 0):
            if (wanted_key - 1 < len(zdiff)):
                numu_wanted_list.append(zdiff[wanted_key])
            else:
                numu_wanted_list.append(zdiff[-1])
        
        #print("")
        #print("ONE DIFF")
        #print(onelist)
        onediff = [abs(k-l) for k,l in zip(onelist[:-1], onelist[1:])]
        #print(onediff) 
        if (len(onelist) != 0):
            if (wanted_key - 1 < len(onediff)):
                numu_wanted_list.append(onediff[wanted_key])
            else:
                numu_wanted_list.append(onediff[-1])
                
        #print("")
        #print("TWO DIFF")
        #print(twolist)
        twodiff = [abs(k-l) for k,l in zip(twolist[:-1], twolist[1:])]
        #print(twodiff)
        if (len(twolist) != 0):
            if (wanted_key - 1 < len(twodiff)):
                numu_wanted_list.append(twodiff[wanted_key])
            else:
                numu_wanted_list.append(twodiff[-1])
        
        #print("")
        #print("THREE DIFF")
        #print(threelist)
        threediff = [abs(k-l) for k,l in zip(threelist[:-1], threelist[1:])]
        #print(threediff)
        if (len(threelist) != 0):
            if (wanted_key - 1 < len(threediff)):
                numu_wanted_list.append(threediff[wanted_key])
            else:
                numu_wanted_list.append(threediff[-1])
        
        #print("")
        #print("FOUR DIFF")
        #print(fourlist)
        fourdiff = [abs(k-l) for k,l in zip(fourlist[:-1], fourlist[1:])]
        #print(fourdiff) 
        if (len(fourlist) != 0):
            if (wanted_key - 1 < len(fourdiff)):
                numu_wanted_list.append(fourdiff[wanted_key])
            else:
                numu_wanted_list.append(fourdiff[-1])
        
        
        #print("")
        #print("FIVE DIFF")
        #print(fivelist)
        fivediff = [abs(k-l) for k,l in zip(fivelist[:-1], fivelist[1:])]
        #print(fivediff)
        if (len(fivelist) != 0):
            if (wanted_key - 1 < len(fivediff)):
                numu_wanted_list.append(fivediff[wanted_key])
            else:
                numu_wanted_list.append(fivediff[-1])
        
        
        #print("")
        #print("SIX DIFF")
        #print(sixlist)
        sixdiff = [abs(k-l) for k,l in zip(sixlist[:-1], sixlist[1:])]
        #print(sixdiff)
        if (len(sixlist) != 0):
            if (wanted_key - 1 < len(sixdiff)):
                numu_wanted_list.append(sixdiff[wanted_key])
            else:
                numu_wanted_list.append(sixdiff[-1])
        
        
        #print("")
        #print("SEVEN DIFF")
        #print(sevenlist)
        sevendiff = [abs(k-l) for k,l in zip(sevenlist[:-1], sevenlist[1:])]
        #print(sevendiff)
        if (len(sevenlist) != 0):
            if (wanted_key - 1 < len(sevendiff)):
                numu_wanted_list.append(sevendiff[wanted_key])
            else:
                numu_wanted_list.append(sevendiff[-1])
        
        
        #print("")
        print("EIGHT DIFF")
        print(eightlist)
        eightdiff = [abs(k-l) for k,l in zip(eightlist[:-1], eightlist[1:])]
        print(eightdiff)
        if (len(eightlist) != 0):
            if (wanted_key - 1 < len(eightdiff)):
                if (eightdiff[wanted_key] > 0.00001):
                    numu_wanted_list.append(eightdiff[wanted_key])
                else:
                    numu_wanted_list.append(0.0) 
            else:
                numu_wanted_list.append(eightdiff[-1])
                
        #print("")
        #print("NINE DIFF")
        #print(ninelist)
        ninediff = [abs(k-l) for k,l in zip(ninelist[:-1], ninelist[1:])]
        #print(ninediff)
        if (len(ninelist) != 0):
            if (wanted_key - 1 < len(ninediff)):
                if (ninediff[wanted_key] > 0.00001):
                    numu_wanted_list.append(ninediff[wanted_key])
                else:
                    numu_wanted_list.append(0.0) 
            else:
                if (ninediff[-1] > 0.00001):
                    numu_wanted_list.append(ninediff[-1])
                else:
                    numu_wanted_list.append(0.0)
                
        print("") 
        print("SUMS")
        print(sumlist)
        difflist = [abs(k-l) for k,l in zip(sumlist[:-1], sumlist[1:])]
        print(difflist)
        print("")
        print("Numu Wanted List:")
        print(numu_wanted_list)
        print("")
        
        return numu_wanted_list