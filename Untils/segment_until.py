import torch
import torch.nn.functional as F

def Accuracy(net,dataloader,loss_function,device,crop_size):
    loss_get=0
    loss=[]
    total=0
    correct=0
    net.eval()
    name,label,target,predict=[],[],[],[]
    number_pixes=crop_size[0]*crop_size[1]
    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            inputs,labels,image_name=data[0].to(device),data[1].to(device),data[2]
            net=net.to(device)
            outputs=net(inputs)
            _,predicted=torch.max(outputs,1)
            #将label和概率添加进列表中去
            for lp in range(len(labels)):
                name.append(image_name[lp])
                #label.append(labels[lp])
                #target.append(F.softmax(outputs[lp], dim=0)[predicted[lp]])
                #predict.append(predicted[lp])

            # print(predicted,labels)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            #print(predicted,labels)
            temp=loss_function(outputs,labels)
            loss.append(temp)
            loss_get+=temp
        return loss_get/total,correct/(total*number_pixes),loss,name,label,target,predict
'''
计算IOU
'''
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    得到bbox的坐标
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


