#Function to sort the colours in a stacked histogram and order the stacked distributions

def sortStackDists(stacksort, nue_var_dict, nue_weight_dict):
    nue_order_dict = {}
    nue_order_var_dict    = {}
    nue_order_weight_dict = {}
    print("1")
    if (stacksort >= 1 and stacksort <= 3):
        print("2")
        #print("In stacksort 1")
        # figure out ordering based on total yield.
        # Options are to have no exceptions (stacksort=1),   
        # put eLEE on top (stacksort=2), or put nue+eLEE on top (stacksort=3)
        # put numu on top (stacksort >= 4)
        has1   = False
        has10  = False
        has11  = False
        has111 = False
        has12  = False
        for c in nue_var_dict.keys():
            if stacksort >= 2:
                if int(c)==111:
                    has111 = True
                    continue
            if stacksort == 3:
                if int(c)==1:
                    has1 = True
                    continue
                if int(c)==12:
                    has12 = True
                    continue
                    
                if int(c)==10:
                    has10 = True
                    continue
                if int(c)==11:
                    has11 = True
                    continue
            nue_order_dict[c] = sum(nue_weight_dict[c])
            nue_order_dict = {k: v for k, v in sorted(nue_order_dict.items(), key=lambda item: item[1])}
        if has1:
            nue_order_dict[1] = sum(nue_weight_dict[1])
        if has1:
            nue_order_dict[12] = sum(nue_weight_dict[12])
        if has10:
            nue_order_dict[10] = sum(nue_weight_dict[10])
        if has11:
            nue_order_dict[11] = sum(nue_weight_dict[11])
        if has111:
            nue_order_dict[111] = sum(nue_weight_dict[111])
        # now that the order has been sorted out, fill the actual dicts
        print("3")
        for c in nue_order_dict.keys():
            nue_order_var_dict[c] = nue_var_dict[c]
        for c in nue_order_dict.keys():
            nue_order_weight_dict[c] = nue_weight_dict[c]
            #print("order w sum ", sum(nue_order_weight_dict[c]), " c ", c)
    elif stacksort == 4:
        #print("in elif")
        #put the numu stuff on top
        hasprotons = 23 in nue_var_dict.keys()
        keys = list(nue_var_dict.keys())
        if hasprotons:
            keys.remove(22)#take them out
            keys.remove(23)
            keys.remove(24)
            keys.remove(25)
            keys.append(22)#and put at end
            keys.append(23)
            keys.append(24)
            keys.append(25)

        for c in keys:
            nue_order_var_dict[c] = nue_var_dict[c]
            nue_order_weight_dict[c] = weight_dict[c]
    else:
        print("in else")
        for c in nue_var_dict.keys():
            nue_order_var_dict[c] = nue_var_dict[c]
        for c in weight_dict.keys():
            nue_order_weight_dict[c] = weight_dict[c]
            
    try: c
    except NameError: c = 0
        
    #print("c = ", c)        
    return c, nue_order_var_dict, nue_order_weight_dict